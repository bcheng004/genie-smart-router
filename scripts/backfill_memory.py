"""Explicit-profile export of an actor's existing conversations to the memory outbox."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--volume", required=True)
    args = parser.parse_args()
    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    os.environ["LAKEBASE_ENDPOINT"] = args.endpoint
    os.environ["LONG_TERM_MEMORY_VOLUME"] = args.volume

    import long_term_memory
    import memory
    from memory_pipeline.events import source_events
    from memory_pipeline.store import enqueue, ensure_tables

    long_term_memory.source_path()
    chats = memory.load_chats(args.actor)

    def backfill(conn):
        ensure_tables(conn, memory.SCHEMA)
        for chat in chats:
            chat["id"] = chat["chat_id"]
            enqueue(conn, memory.SCHEMA, source_events(args.actor, chat))

    memory._run(backfill)
    published = 0
    while count := long_term_memory.publish_pending(actor_id=args.actor):
        published += count
    print(f"Queued {len(chats)} chats; published {published} source events.")


if __name__ == "__main__":
    main()
