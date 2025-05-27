import os
import sys
from dotenv import load_dotenv
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import glob

# Load environment variables from the .env file
load_dotenv(dotenv_path=".env.micro.trading")

# Configuration for DigitalOcean Spaces
OBJECT_STORAGE_URL = os.getenv("OBJECT_STORAGE_URL")  # Your DigitalOcean endpoint URL
ACCESS_KEY = os.getenv("ACCESS_KEY")  # Replace with your DigitalOcean Spaces access key
SECRET_KEY = os.getenv("SECRET_KEY")  # Replace with your DigitalOcean Spaces secret key
BUCKET_NAME = os.getenv("BUCKET_NAME")  # Your bucket name
REGION_NAME = os.getenv("REGION_NAME")  # Your region name

# Initialize the S3 client with explicit region configuration
s3_client = boto3.client(
    's3', 
    endpoint_url=OBJECT_STORAGE_URL,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=REGION_NAME  # Ensure region is set
)

def download_model(bucket_name, key, local_path):
    """Download a file from DigitalOcean Spaces."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        # If the file exists, delete it first
        if os.path.exists(local_path):
            os.remove(local_path)
        s3_client.download_file(bucket_name, key, local_path)
        
        return True
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except PartialCredentialsError:
        print("Incomplete credentials provided")
        return False
    except Exception as e:
        print(f"Model not found or error downloading: {e}")
        return False

def upload_model(bucket_name, key, local_path):
    """Upload a file to DigitalOcean Spaces."""
    try:
        s3_client.upload_file(local_path, bucket_name, key)
        print(f"Model uploaded to {bucket_name}/{key}")
    except Exception as e:
        print(f"Error uploading model: {e}")