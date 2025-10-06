# inference script
import boto3
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError
import os
import pandas as pd
import mlflow


def upload_file_to_s3(
    file_path,
    bucket_name,
    object_key=None,
    endpoint_url=None,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    region="us-east-1",
):
    """
    Upload a file to S3-compatible storage.

    Args:
        file_path (str): Path to the file to upload
        bucket_name (str): Name of the S3 bucket
        object_key (str, optional): S3 object key. If not specified,
            uses filename
        endpoint_url (str, optional): Custom endpoint URL for
            S3-compatible services
        aws_access_key_id (str, optional): AWS access key ID
        aws_secret_access_key (str, optional): AWS secret access key
        region (str): AWS region (default: us-east-1)

    Returns:
        bool: True if file was uploaded successfully, False otherwise
    """

    # Validate file exists
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        return False

    # Use filename as object key if not specified
    if object_key is None:
        object_key = Path(file_path).name

    try:
        # Create S3 client
        s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region,
        )

        # Get file size for progress tracking
        file_size = os.path.getsize(file_path)
        print(
            f"Uploading '{file_path}' ({file_size} bytes) to bucket "
            f"'{bucket_name}' as '{object_key}'..."
        )

        # Upload the file
        s3_client.upload_file(
            file_path,
            bucket_name,
            object_key,
            # Optional: server-side encryption
            # ExtraArgs={"ServerSideEncryption": "AES256"},
        )

        print(f"✅ Successfully uploaded '{object_key}' to bucket " f"'{bucket_name}'")

        # Generate the file URL (for AWS S3)
        if endpoint_url is None:
            file_url = (
                f"https://{bucket_name}.s3.{region}." f"amazonaws.com/{object_key}"
            )
        else:
            file_url = f"{endpoint_url}/{bucket_name}/{object_key}"

        print(f"📁 File URL: {file_url}")
        return True

    except NoCredentialsError:
        print(
            "❌ Error: AWS credentials not found. " "Please configure your credentials."
        )
        print("You can set them via:")
        print("  - Environment variables " "(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        print("  - AWS credentials file (~/.aws/credentials)")
        print("  - IAM roles (if running on EC2)")
        return False

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "NoSuchBucket":
            print(f"❌ Error: Bucket '{bucket_name}' does not exist.")
        elif error_code == "AccessDenied":
            print(
                f"❌ Error: Access denied. Check your permissions for "
                f"bucket '{bucket_name}'."
            )
        else:
            print(f"❌ Error uploading file: {e}")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def download_file_from_s3(
    bucket_name,
    object_key,
    file_path=None,
    endpoint_url=None,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    region="us-east-1",
):
    """
    Download a file from S3-compatible storage.

    Args:
        bucket_name (str): Name of the S3 bucket
        object_key (str): S3 object key to download
        file_path (str, optional): Local path to save the file. If not
            specified, uses object key filename in current directory
        endpoint_url (str, optional): Custom endpoint URL for
            S3-compatible services
        aws_access_key_id (str, optional): AWS access key ID
        aws_secret_access_key (str, optional): AWS secret access key
        region (str): AWS region (default: us-east-1)

    Returns:
        bool: True if file was downloaded successfully, False otherwise
    """

    # Use object key filename as local file path if not specified
    if file_path is None:
        file_path = Path(object_key).name

    try:
        # Create S3 client
        s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region,
        )

        # Check if object exists and get its size
        try:
            response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
            file_size = response["ContentLength"]
            print(
                f"Downloading '{object_key}' ({file_size} bytes) from bucket "
                f"'{bucket_name}' to '{file_path}'..."
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print(
                    f"❌ Error: Object '{object_key}' not found in bucket "
                    f"'{bucket_name}'."
                )
                return False
            else:
                raise Exception("Error checking object existence") from e

        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Download the file
        s3_client.download_file(bucket_name, object_key, file_path)

        print(f"✅ Successfully downloaded '{object_key}' to '{file_path}'")

        # Verify file was downloaded and show size
        if os.path.isfile(file_path):
            local_size = os.path.getsize(file_path)
            print(f"📁 Local file size: {local_size} bytes")

        return True

    except NoCredentialsError:
        print(
            "❌ Error: AWS credentials not found. " "Please configure your credentials."
        )
        print("You can set them via:")
        print("  - Environment variables " "(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        print("  - AWS credentials file (~/.aws/credentials)")
        print("  - IAM roles (if running on EC2)")
        return False

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "NoSuchBucket":
            print(f"❌ Error: Bucket '{bucket_name}' does not exist.")
        elif error_code == "AccessDenied":
            print(
                f"❌ Error: Access denied. Check your permissions for "
                f"bucket '{bucket_name}' and object '{object_key}'."
            )
        else:
            print(f"❌ Error downloading file: {e}")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def predict(input_data):
    logged_model = os.getenv("MLFLOW_LOGGED_MODEL")

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict on a Pandas DataFrame.
    return loaded_model.predict(input_data)


def persist_predictions(df):
    df.to_csv("/tmp/predictions.csv", index=False)
    uploaded = upload_file_to_s3(
        bucket_name=os.getenv("S3_BUCKET_NAME", None),
        object_key="data/predictions.csv",
        file_path="/tmp/predictions.csv",
        endpoint_url=os.getenv("S3_ENDPOINT_URL", None),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", None),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", None),
        region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    return uploaded


if __name__ == "__main__":
    # Example usage
    bucket_name = os.getenv("S3_BUCKET_NAME", None)
    object_key = os.getenv("S3_OBJECT_KEY", None)
    endpoint_url = os.getenv("S3_ENDPOINT_URL", None)
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", None)
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", None)
    aws_default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    downloaded = download_file_from_s3(
        bucket_name,
        object_key,
        file_path="/tmp/test.csv",
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region=aws_default_region,
    )

    if downloaded:
        df = pd.read_csv("/tmp/test.csv")
        print(df.head())  # Display the first few rows of the DataFrame

    else:
        print("File download failed.")

    mlflow_server_url = os.getenv("MLFLOW_TRACKING_URI")

    # MLFlow setup
    # Set tracking URI
    mlflow.set_tracking_uri(mlflow_server_url)
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    # predict
    try:
        print("Making predictions...")
        predictions = predict(df)
    except Exception as e:
        raise Exception(f"❌ Error during prediction: {e}")

    preds_df = pd.DataFrame(
        {
            "PassengerId": df["PassengerId"],
            "predictions": predictions,
        }
    )

    print("Predictions:")
    print(preds_df.head())
    persist_predictions(preds_df)
