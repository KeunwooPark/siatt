"""Standing tasks: a schedule somebody created, delivered as an ordinary turn.

Every recurring thing Siatt did before this was compiled in. `default_specs`
(`siatt/runner/jobs.py`) is a list written by whoever built the binary, and the
clock iterates exactly that list — so there was no way for a person to say
"every weekday at nine, tell me what happened in AI overnight" and have it
happen. A standing task is that: a row someone created, fired by the daemon,
answered in the thread they created it in.

The interesting claim in this module is how little it adds.

* The clock already queues occurrences under an id derived from their fire
  time (`scheduled_id`), which is what makes it idempotent, safe to run twice,
  and free of a stampede after downtime. A task's occurrences use the same
  trick, so none of that is re-reasoned here.
* The handler does **not** run the agent. It builds an `InboundEvent` and puts
  it in the inbox. From there it is an ordinary turn — dispatcher, session
  actor, context packing, retrieval, tools, and the surface's own reply path —
  and no adapter needs to learn anything: Slack answers from `event.channel`
  and `event.reply_to` alone.
* `UNIQUE (source, external_id)` on the inbox is what stops a *retried job*
  producing a second answer. The job's own id is the event's `external_id`, so
  the two queues' at-least-once semantics compose into at-most-one-answer
  without a new mechanism.

A task may also answer somewhere other than the thread it was asked in, and
the shape of that is the whole of `Delivery` below. What a row holds is a
destination *name*, never a channel: `here` for the conversation it was created
in, at top level so each firing starts its own thread, or a key in
`[tasks.destinations]`, which is operator config and the only place a channel
id exists. Nothing arriving in a conversation can name a channel that is not
already that conversation (§7.1), and a firing runs under the visibility of
where it lands rather than of who asked for it (§11.1).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Self
from zoneinfo import ZoneInfo

from ulid import ULID

from siatt.adapters.slack.events import scope_for
from siatt.config import HERE, TaskSettings
from siatt.core.events import InboundEvent
from siatt.core.inbox import Inbox
from siatt.errors import SiattError
from siatt.runner.cron import Cron, CronError
from siatt.runner.scheduler import Job, JobHandler, Occurrence
from siatt.store import Store

log = logging.getLogger(__name__)

#: The one job kind standing tasks run under. Registered with no cron of its
#: own: the occurrences come from the tasks table, and this exists so the queue
#: accepts the kind and the drainer has a handler for it.
TASK_KIND = "task_run"

ACTIVE, PAUSED, DONE = "active", "paused", "done"

#: How a paused task's owner is told. None where there is nothing to tell them
#: with — a build with no Slack, or a task created from a terminal.
TaskNotifier = Callable[["Task", str], Awaitable[None]]


class TaskError(SiattError):
    """A schedule Siatt will not create, or one it can no longer read."""


def occurrence_id(task_id: str, fire_at: datetime) -> str:
    """The job id of one firing.

    Derived from the fire time for the same reason `scheduled_id` is: two
    schedulers ticking on the same minute write the same row, and only the
    first one counts. Prefixed with `task:` so a glance at the jobs table says
    which task a row belongs to.
    """
    return f"task:{task_id}@{fire_at.isoformat(timespec='minutes')}"


@dataclass(frozen=True, slots=True)
class Task:
    """One standing schedule, as it is stored."""

    id: str
    owner: str
    surface: str
    session_id: str
    channel: str | None
    reply_to: str | None
    scope: str
    prompt: str
    cron: str
    timezone: str | None
    state: str
    fire_once: bool
    #: Where each firing is answered, as a name: None is the thread it was
    #: created in, `here` is that channel at top level, anything else is a key
    #: in `[tasks.destinations]` that only config can resolve.
    destination: str | None
    created_at: str
    last_run_at: str | None = None
    last_job_id: str | None = None
    last_error: str | None = None
    consecutive_failures: int = 0

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Self:
        return cls(
            id=str(row["id"]),
            owner=str(row["owner"]),
            surface=str(row["surface"]),
            session_id=str(row["session_id"]),
            channel=row["channel"],
            reply_to=row["reply_to"],
            scope=str(row["scope"]),
            prompt=str(row["prompt"]),
            cron=str(row["cron"]),
            timezone=row["timezone"],
            state=str(row["state"]),
            fire_once=bool(row["fire_once"]),
            destination=row["destination"],
            created_at=str(row["created_at"]),
            last_run_at=row["last_run_at"],
            last_job_id=row["last_job_id"],
            last_error=row["last_error"],
            consecutive_failures=int(row["consecutive_failures"]),
        )

    def schedule(self) -> Cron:
        """The parsed expression. Raises if the row no longer reads.

        It can stop reading: the zone is resolved against whatever tz database
        the machine has, and a task created on one host and run on another can
        name a zone the second one has never heard of.
        """
        return Cron.parse(self.cron, tz=self.timezone)

    def next_fires(self, count: int = 3, *, now: datetime | None = None) -> list[datetime]:
        """The next `count` instants this fires on, earliest first.

        What confirmation is built out of. Nobody can check `0 9 * * 1-5`, and
        everybody can check "Mon 8 Sep 09:00, Tue 9 Sep 09:00".
        """
        schedule = self.schedule()
        moment = now or datetime.now(UTC)
        fires = []
        for _ in range(count):
            moment = schedule.next_after(moment)
            fires.append(moment)
        return fires

    def delivery(self, job_id: str, destinations: Mapping[str, str]) -> Delivery:
        """Where this firing is answered, resolved for the run about to happen.

        At fire time rather than at creation, and that is the point: the row
        names a destination and only `[tasks.destinations]` can say what the
        name means, so where a task posts is a thing the operator holds and
        keeps holding. A name that is no longer configured resolves to nothing
        and raises — the run fails, the task is paused after enough of them and
        its owner is told, which is the one safe reading of "post this
        somewhere I can no longer find".
        """
        if self.destination is None:
            return Delivery(
                session_id=self.session_id,
                channel=self.channel,
                reply_to=self.reply_to,
                scope=self.scope,
            )
        if self.destination == HERE:
            if not self.channel:
                raise TaskError(
                    f"task {self.id} posts in the channel it was created in, and it has none"
                )
            channel, scope = self.channel, self.scope
        elif resolved := destinations.get(self.destination):
            channel = resolved
            # The destination's own scope, never the creator's. A task set up
            # from a DM and pointed at a channel answers with what that channel
            # may see and nothing else — the leak §11.1 exists to prevent is
            # exactly the one where a private scope travels to a public place.
            scope = scope_for(channel, self.owner, is_dm=False)
        else:
            raise TaskError(
                f"task {self.id} posts to {self.destination!r}, which is not a destination in "
                "this config; add it to [tasks.destinations] or cancel the task"
            )
        return Delivery(
            # One firing, one conversation. Derived from the job id, which is
            # derived from the fire time, so a retried run rejoins the same
            # conversation instead of opening a second one beside it — and
            # prefixed with the surface, because that is where everything else
            # reads the surface from.
            session_id=f"{self.surface}:{job_id}",
            channel=channel,
            # A message with no thread is a thread: this is what makes each
            # firing its own post rather than another line in an old one.
            reply_to=None,
            scope=scope,
        )

    @property
    def label(self) -> str:
        """How to name this schedule in a listing or a log line.

        Formatted rather than parsed. `Cron.label` would say the same thing,
        but it can only say it about an expression that still reads — and the
        one listing that has to name a task whose zone this machine has never
        heard of is the listing where that has gone wrong.
        """
        return self.cron if self.timezone is None else f"{self.cron} ({self.timezone})"


@dataclass(frozen=True, slots=True)
class Delivery:
    """Where one firing of a task is answered, and under whose visibility.

    Everything a `Task` row is about is history and schedule; this is the part
    the event carries. It exists as a type because the four fields have to be
    decided together — a channel with the wrong scope beside it is the bug
    §11.1 is about, and a scope with no channel is a task that says nothing.
    """

    session_id: str
    channel: str | None
    reply_to: str | None
    scope: str


class Tasks:
    """The tasks table, and every judgement about what may go in it.

    The store holds rows and counts them; what a schedule is *allowed* to be
    lives here, in one place, so the CLI and the `schedule_*` tools cannot
    disagree about it.
    """

    def __init__(self, store: Store, settings: TaskSettings | None = None) -> None:
        self._store = store
        self._settings = settings or TaskSettings()

    @property
    def settings(self) -> TaskSettings:
        return self._settings

    async def create(
        self,
        *,
        owner: str,
        surface: str,
        session_id: str,
        prompt: str,
        cron: str,
        timezone: str | None = None,
        channel: str | None = None,
        reply_to: str | None = None,
        scope: str = "workspace",
        fire_once: bool = False,
        destination: str | None = None,
        now: datetime | None = None,
    ) -> Task:
        """Create a schedule, having checked it is one Siatt will honour."""
        if not prompt.strip():
            raise TaskError("a task needs something to do; the prompt is empty")
        if destination is not None:
            self._validate_destination(destination, surface=surface, channel=channel)
        schedule = self._validate(cron, timezone, now=now)
        if (held := await self._store.count_owner_tasks(owner)) >= self._settings.max_per_owner:
            raise TaskError(
                f"{owner} already has {held} schedule(s), which is the limit "
                f"({self._settings.max_per_owner}). Cancel one first."
            )
        task_id = str(ULID())
        await self._store.create_task(
            task_id=task_id,
            owner=owner,
            surface=surface,
            session_id=session_id,
            channel=channel,
            reply_to=reply_to,
            scope=scope,
            prompt=prompt.strip(),
            cron=schedule.expression,
            timezone=timezone,
            fire_once=fire_once,
            destination=destination,
        )
        created = await self.get(task_id)
        if created is None:  # pragma: no cover - the row was just written
            raise TaskError(f"task {task_id} vanished between writing and reading it")
        return created

    def _validate_destination(self, destination: str, *, surface: str, channel: str | None) -> None:
        """Refuse a destination now, rather than at nine tomorrow morning.

        Checked here and resolved again at fire time, which is not a
        duplicated check: this one is "is that a thing you can ask for", and
        that one is "what does the name mean today". Only the second can change
        after the task exists, and only the first can be got wrong by the
        person setting it up.
        """
        if surface != "slack":
            raise TaskError(f"a destination is a Slack channel, and this task runs on {surface!r}")
        if destination == HERE:
            if not channel:
                raise TaskError("there is no channel here to post in")
            return
        if destination not in self._settings.destinations:
            named = ", ".join(sorted(self._settings.destinations)) or "none are configured"
            raise TaskError(
                f"there is no destination named {destination!r} ({named}). "
                "Destinations are channels the operator names in [tasks.destinations]."
            )

    def _validate(self, cron: str, timezone: str | None, *, now: datetime | None) -> Cron:
        """Parse the expression, and refuse one that fires too often.

        The floor is measured on the gap the expression actually produces
        rather than on how it is written, so `*/15` and `0,15,30,45` are the
        same schedule and are judged the same way.
        """
        try:
            schedule = Cron.parse(cron, tz=timezone)
        except CronError as exc:
            raise TaskError(str(exc)) from exc
        moment = now or datetime.now(UTC)
        try:
            first = schedule.next_after(moment)
            gap = (schedule.next_after(first) - first).total_seconds() / 60
        except CronError as exc:
            raise TaskError(str(exc)) from exc
        if gap < self._settings.min_interval_minutes:
            raise TaskError(
                f"{schedule.label} fires every {gap:.0f} minute(s), and the floor is "
                f"{self._settings.min_interval_minutes}. Every fire is a full turn."
            )
        return schedule

    async def get(self, task_id: str) -> Task | None:
        row = await self._store.get_task(task_id)
        return Task.from_row(row) if row else None

    async def all(
        self, *, state: str | None = None, owner: str | None = None, session_id: str | None = None
    ) -> list[Task]:
        """Every task matching the narrowing, oldest first.

        Not named `list`: a method by that name shadows the builtin inside the
        class body, and every `list[Task]` annotation below it stops meaning a
        list.
        """
        return [
            Task.from_row(row)
            for row in await self._store.list_tasks(state=state, owner=owner, session_id=session_id)
        ]

    async def cancel(self, task_id: str) -> bool:
        """Delete a task. An occurrence already queued for it will not post."""
        return await self._store.delete_task(task_id)

    async def pause(self, task_id: str, *, reason: str | None = None) -> bool:
        return await self._store.set_task_state(task_id, state=PAUSED, error=reason)

    async def resume(self, task_id: str) -> bool:
        return await self._store.set_task_state(task_id, state=ACTIVE)

    async def finish(self, task_id: str) -> bool:
        return await self._store.set_task_state(task_id, state=DONE)

    async def record_failure(self, task_id: str, *, error: str) -> int:
        return await self._store.record_task_failure(task_id, error=error)

    async def occurrences(self, moment: datetime) -> list[Occurrence]:
        """The next firing of every active task. The clock's half of this.

        Per task, because one whose expression or zone no longer reads must not
        stop the tasks behind it — the same isolation the spec loop already
        has, for the same reason, and here it matters more: these expressions
        were written by a model reading what somebody typed.
        """
        found = []
        for task in await self.all(state=ACTIVE):
            try:
                fire_at = task.schedule().next_after(moment)
            except CronError:
                log.exception(
                    "task %s has a schedule that no longer reads (%s)", task.id, task.cron
                )
                continue
            found.append(
                Occurrence(
                    job_id=occurrence_id(task.id, fire_at),
                    kind=TASK_KIND,
                    fire_at=fire_at,
                    payload={"task_id": task.id},
                    label=f"task {task.id} ({task.label})",
                )
            )
        return found


def task_handler(
    store: Store,
    settings: TaskSettings | None = None,
    *,
    inbox: Inbox | None = None,
    notify: TaskNotifier | None = None,
) -> JobHandler:
    """Run one occurrence: put the task's prompt in the inbox as a message.

    Deliberately not "run the agent". Everything that makes a turn a turn —
    one actor per conversation, retrieval, the surface's reply path, the
    failure handling that redelivers a message whose model call fell over —
    already exists on the inbox side of the queue, and a second path to it
    would be a second set of all of that to keep correct.

    What this counts as a failure is therefore narrow, and worth being plain
    about: the enqueue, not the turn. A model that times out answering a
    standing task is an inbox failure with the inbox's own retries, and it is
    not what pauses a task. What pauses a task is the run never reaching the
    inbox at all — a deleted session, a row that stopped parsing, a destination
    nobody configures any more — which is the failure that would otherwise
    repeat silently forever.
    """
    tasks = Tasks(store, settings)
    queue = inbox or Inbox(store)

    async def run(job: Job) -> None:
        task_id = str(job.payload.get("task_id") or "")
        task = await tasks.get(task_id)
        if task is None:
            # Cancelled between the clock queueing this and the drainer
            # reaching it. Deleting a task stops it, so this run is not work
            # that was lost — it is work that was called off.
            log.info("task %s is gone; dropping the run queued for it", task_id or "?")
            return
        if task.state != ACTIVE:
            log.info("task %s is %s; skipping this run", task.id, task.state)
            return
        try:
            await _deliver(queue, task, job, tasks.settings.destinations)
        except Exception as exc:
            # Once per occurrence, not once per attempt. The job retries on the
            # queue's own backoff, and counting each of those three tries as a
            # failed *run* would pause a task after two bad fire times while
            # claiming it had six.
            if job.attempts <= 1:
                await _blame(tasks, task, exc, notify=notify)
            raise
        await store.record_task_run(task.id, job_id=job.id)
        if task.fire_once:
            await tasks.finish(task.id)
        log.info("task %s queued a turn in %s", task.id, task.destination or task.session_id)

    return run


async def _deliver(queue: Inbox, task: Task, job: Job, destinations: Mapping[str, str]) -> None:
    """Hand the task's prompt to the inbox as though somebody had said it.

    `external_id` is the job's id, which is the fire time. A job retried after
    a partial failure re-enqueues the same event id, and the inbox's UNIQUE
    constraint turns that into one answer rather than two.

    The destination is resolved here, on the run, and an unresolvable one
    raises before anything is queued: a firing that cannot say where it goes
    must not go anywhere.
    """
    where = task.delivery(job.id, destinations)
    await queue.enqueue(
        InboundEvent(
            source=task.surface,
            external_id=job.id,
            session_id=where.session_id,
            text=task.prompt,
            scope=where.scope,
            author=task.owner,
            channel=where.channel,
            reply_to=where.reply_to,
            origin="scheduled",
        )
    )


async def _blame(tasks: Tasks, task: Task, exc: Exception, *, notify: TaskNotifier | None) -> None:
    """Count a failed run, and stop the task once it has failed enough.

    Told once, on the run that crosses the threshold, and never again: the
    task is paused by the same call, so there is no second crossing to
    announce. A notifier that itself fails is logged and swallowed — a task
    that could not be paused because nobody could be told would keep failing,
    which is the outcome this exists to prevent.
    """
    reason = f"{type(exc).__name__}: {exc}"
    failures = await tasks.record_failure(task.id, error=reason)
    if failures < tasks.settings.disable_after_failures:
        return
    await tasks.pause(task.id, reason=reason)
    log.error("task %s paused after %d consecutive failures: %s", task.id, failures, reason)
    if notify is None:
        return
    try:
        await notify(
            task,
            f"I have paused a scheduled task after {failures} failed run(s) — "
            f"{task.prompt!r} ({task.label}). The last error was: {reason}",
        )
    except Exception:
        log.exception("could not tell %s that task %s was paused", task.owner, task.id)


def render_fires(fires: Sequence[datetime], tz_name: str | None) -> list[str]:
    """Fire times as somebody reads them: in the task's own zone, named.

    UTC instants are what the scheduler deals in and are useless as
    confirmation — "2026-09-08T00:00:00+00:00" is not something a person in
    Seoul can check against "every weekday at nine".
    """
    zone = ZoneInfo(tz_name) if tz_name else UTC
    suffix = tz_name or "UTC"
    return [f"{fire.astimezone(zone).strftime('%a %d %b %H:%M')} {suffix}" for fire in fires]
