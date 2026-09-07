"""Pluggable event transport.

The pipeline is written against :class:`~apexpulse.broker.base.EventBroker`, never
against a concrete client, so the same producer and consumer code runs on an
in-process queue during development and on Redpanda in production.
"""

from apexpulse.broker.base import BrokerMessage, EventBroker
from apexpulse.broker.factory import create_broker
from apexpulse.broker.memory import InMemoryBroker

__all__ = ["BrokerMessage", "EventBroker", "InMemoryBroker", "create_broker"]
