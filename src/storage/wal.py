from decimal import Decimal
import boto3
from boto3.dynamodb.conditions import Key

from config import DYNAMODB_REGION, DYNAMODB_ENDPOINT, WAL_TABLE, META_TABLE, QUERY_NODE_ID


class WALClient:
    def __init__(self):
        db = boto3.resource(
            "dynamodb",
            region_name=DYNAMODB_REGION,
            endpoint_url=DYNAMODB_ENDPOINT,
            aws_access_key_id="local",
            aws_secret_access_key="local",
        )
        self._wal = db.Table(WAL_TABLE)
        self._meta = db.Table(META_TABLE)

    def next_vector_id(self) -> int:
        resp = self._meta.update_item(
            Key={"counter_id": "global"},
            UpdateExpression="ADD next_id :one",
            ExpressionAttributeValues={":one": 1},
            ReturnValues="UPDATED_NEW",
        )
        return int(resp["Attributes"]["next_id"])

    def write(self, seq_id: int, operation: str, vector=None, vector_id=None) -> None:
        item = {
            "Query_Node_ID": QUERY_NODE_ID,
            "seq_id":        seq_id,
            "operation":     operation,
        }
        if vector is not None:
            item["vector"] = [Decimal(str(x)) for x in vector]
        if vector_id is not None:
            item["vector_id"] = str(vector_id)
        self._wal.put_item(Item=item)

    def read_after(self, last_seq: int) -> list[dict]:
        kwargs = dict(
            KeyConditionExpression=(
                Key("Query_Node_ID").eq(QUERY_NODE_ID) &
                Key("seq_id").gt(last_seq)
            )
        )
        items = []
        while True:
            resp = self._wal.query(**kwargs)
            items.extend(resp["Items"])
            if "LastEvaluatedKey" not in resp:
                break
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        return sorted(items, key=lambda x: int(x["seq_id"]))
