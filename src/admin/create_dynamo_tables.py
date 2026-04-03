import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from botocore.exceptions import ClientError

from config import DYNAMODB_REGION, DYNAMODB_ENDPOINT, WAL_TABLE, META_TABLE


def create_table(client, name, key_schema, attribute_definitions):
    try:
        client.create_table(
            TableName=name,
            KeySchema=key_schema,
            AttributeDefinitions=attribute_definitions,
            BillingMode="PAY_PER_REQUEST",
        )
        print(f"  Created table: {name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            print(f"  Table already exists: {name}")
        else:
            raise


def main():
    client = boto3.client(
        "dynamodb",
        region_name=DYNAMODB_REGION,
        endpoint_url=DYNAMODB_ENDPOINT,
        aws_access_key_id="local",
        aws_secret_access_key="local",
    )

    # WAL table: PK=Query_Node_ID, SK=seq_id
    create_table(
        client,
        WAL_TABLE,
        key_schema=[
            {"AttributeName": "Query_Node_ID", "KeyType": "HASH"},
            {"AttributeName": "seq_id",        "KeyType": "RANGE"},
        ],
        attribute_definitions=[
            {"AttributeName": "Query_Node_ID", "AttributeType": "S"},
            {"AttributeName": "seq_id",        "AttributeType": "N"},
        ],
    )

    # VectorIndexMeta table: single global item holding the next_id counter
    create_table(
        client,
        META_TABLE,
        key_schema=[
            {"AttributeName": "counter_id", "KeyType": "HASH"},
        ],
        attribute_definitions=[
            {"AttributeName": "counter_id", "AttributeType": "S"},
        ],
    )

    print("Done.")


if __name__ == "__main__":
    main()
