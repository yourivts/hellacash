"""Tests for bot.events.bus.EventBus."""
from __future__ import annotations

import asyncio

import pytest

from bot.events.bus import EventBus


@pytest.mark.asyncio
async def test_publish_subscribe():
    bus = EventBus()
    queue = await bus.subscribe("test.topic")
    await bus.publish("test.topic", {"value": 42})

    event = queue.get_nowait()
    assert event.topic == "test.topic"
    assert event.payload == {"value": 42}


@pytest.mark.asyncio
async def test_multiple_subscribers():
    bus = EventBus()
    q1 = await bus.subscribe("test.topic")
    q2 = await bus.subscribe("test.topic")

    await bus.publish("test.topic", {"msg": "hello"})

    e1 = q1.get_nowait()
    e2 = q2.get_nowait()
    assert e1.payload == {"msg": "hello"}
    assert e2.payload == {"msg": "hello"}


@pytest.mark.asyncio
async def test_queue_full_drops_event():
    bus = EventBus()
    queue = await bus.subscribe("test.topic", maxsize=1)

    # Fill the queue
    await bus.publish("test.topic", {"n": 1})
    # This should be dropped (queue full), not raise
    await bus.publish("test.topic", {"n": 2})

    assert queue.qsize() == 1
    event = queue.get_nowait()
    assert event.payload == {"n": 1}
