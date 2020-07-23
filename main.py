import os
import time
import json
import boto3 as boto3
import facerecognition_client
from botocore.config import Config
from facerecognition_client import Configuration, Locator, ExtractFacesInput, \
    Job, JobMediatorInput, Token, AuthApi, JobsApi, \
    MediatorJob, JobMediatorStatus, ExtractFacesOutput, \
    ClusterFacesInput, ClusterFacesOutput, \
    SearchFacesInput, SearchFacesOutput

from facerecognition_client.rest import ApiException
from pprint import pprint

# Read configuration from json file.
with open(os.environ['APP_CONFIG_FILE']) as json_file:
    config_file = json.load(json_file)

mediator_config = Configuration()
if 'host' in config_file:
    mediator_config.host = config_file['host']


client = config_file['clientKey']
secret = config_file['clientSecret']
project_service_id = config_file['projectServiceId']

aws_access_key_id = config_file['aws_access_key_id']
aws_secret_access_key = config_file['aws_secret_access_key']
region_name = config_file['bucketRegion']

#aws_session_token = config_file['aws_session_token']

# Create an S3 client
# https://github.com/boto/boto3/issues/1644
# This is very important to initialize s3 client with region name, addressing_style & signature_version
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name, config=Config(s3={'addressing_style': 'path'}, signature_version='s3v4'))

# input & output data
bucketName = config_file['bucketName']

inputFile = config_file['localPath'] + config_file['inputFile']
inputKey = config_file['inputFile']
inputImageKey = config_file['inputImage']
inputImageFile = config_file['localPath'] + config_file['inputImage']

def upload_media_to_s3():
    pprint('Uploading media to S3 ...')
    s3.upload_file(inputFile, bucketName, inputKey)
    pprint('Media was uploaded to s3')

def upload_image_to_s3():
    pprint('Uploading image to S3 ...')
    s3.upload_file(inputImageFile, bucketName, inputImageKey)
    pprint('Image was uploaded to s3')



def download_result_from_s3():
    pprint('Downloading result from S3 ...')
    s3.download_file(bucketName, outputKey_json, outputFile_json)
    s3.download_file(bucketName, outputKey_ttml, outputFile_ttml)
    s3.download_file(bucketName, outputKey_text, outputFile_text)
    pprint('Result was downloaded from s3')

def delete_artifacts_from_s3():
    pprint('Deleting s3 artifacts ...')
    # s3.delete_object(Bucket=bucketName, Key=inputKey)
    # s3.delete_object(Bucket=bucketName, Key=outputKey_json)
    # s3.delete_object(Bucket=bucketName, Key=outputKey_ttml)
    # s3.delete_object(Bucket=bucketName, Key=outputKey_text)
    pprint('S3 artifacts were deleted')

def get_signed_url_input():
    return s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={
            'Bucket': bucketName,
            'Key': inputKey
        },
        ExpiresIn=60*60)

def get_signed_image_url_input():
    return s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={
            'Bucket': bucketName,
            'Key': inputImageKey
        },
        ExpiresIn=60*60)



def get_signed_url_output(outputKey):
    return s3.generate_presigned_url(
        ClientMethod='put_object',
        Params={
            'Bucket': bucketName,
            'Key': outputKey
        },
        ExpiresIn=60*60)


def get_access_token() -> Token:
    # create an instance of the API class
    api_instance: AuthApi = facerecognition_client.AuthApi(facerecognition_client.ApiClient(mediator_config))
    return api_instance.get_access_token(client, secret)


def create_extract_faces_job(signed_url_input) -> Job:
    video: Locator = Locator(bucketName, inputKey, signed_url_input)
    job_input: ExtractFacesInput = ExtractFacesInput(video=video, effort='Low')
    return Job(job_type='FaceRecognitionJob', job_profile='ExtractFaces', job_input=job_input)

def create_search_faces_job(signed_url_input, cluster_collection_id) -> Job:
    inputImage: Locator = Locator(bucketName, inputImageKey, signed_url_input)
    job_input: SearchFacesInput = SearchFacesInput(input_image=inputImage, cluster_collection_id=cluster_collection_id, similarity_threshold=0.8)
    return Job(job_type='FaceRecognitionJob', job_profile='SearchFaces', job_input=job_input)


def create_cluster_faces_job(face_extraction_id) -> Job:
    job_input: ClusterFacesInput = ClusterFacesInput(face_extraction_id=face_extraction_id)
    return Job(job_type='FaceRecognitionJob', job_profile='ClusterFaces', job_input=job_input)


def create_job_mediator_input_extract_faces(signed_url_input) -> JobMediatorInput:
    job: Job = create_extract_faces_job(signed_url_input)
    return JobMediatorInput(project_service_id=project_service_id, quantity=6, job=job)

def create_job_mediator_input_search_faces(signed_url_input, cluster_collection_id) -> JobMediatorInput:
    job: Job = create_search_faces_job(signed_url_input, cluster_collection_id)
    return JobMediatorInput(project_service_id=project_service_id, quantity=6, job=job)



def create_job_mediator_input_cluster_faces(face_extraction_id) -> JobMediatorInput:
    job: Job = create_cluster_faces_job(face_extraction_id)
    return JobMediatorInput(project_service_id=project_service_id, quantity=6, job=job)


def submit_job(job_mediator_input) -> MediatorJob:
    # create an instance of the API class
    api_instance: JobsApi = facerecognition_client.JobsApi(facerecognition_client.ApiClient(mediator_config))
    return api_instance.create_job(job_mediator_input)


def get_mediator_job(job_id) -> MediatorJob:
    # create an instance of the API class
    api_instance: JobsApi = facerecognition_client.JobsApi(facerecognition_client.ApiClient(mediator_config))
    return api_instance.get_job_by_id(job_id)


def wait_for_complete(mediator_job):
    mediator_status: JobMediatorStatus = mediator_job.status
    while mediator_status.status not in ["COMPLETED", "FAILED"]:
        time.sleep(30)
        mediator_job = get_mediator_job(mediator_job.id)
        mediator_status = mediator_job.status
        pprint(mediator_job)

    if mediator_status.status == "FAILED":
        raise Exception(mediator_status.status_message)

    return mediator_job


def main():
    try:
        pprint('Mediator host:')
        pprint(mediator_config.host)

        # """
        # Upload to S3
        upload_media_to_s3()
        
        # Get signed urls
        pprint('Receiving signed url ...')
        input_url = get_signed_url_input()

        pprint('input_url:')
        pprint(input_url)

        # Get access token for this client.
        pprint('Retreiving access tokens ...')
        token = get_access_token()

        # Update api_key with access token information for next API calls
        mediator_config.api_key['Authorization'] = token.authorization

        # Create sample job mediator input structure.
        job_mediator_input = create_job_mediator_input_extract_faces(input_url)
        pprint(job_mediator_input)

        # Submit the input job to Mediator
        pprint('Submitting extract faces job ...')
        mediator_job = submit_job(job_mediator_input)
        pprint(job_mediator_input)

        # Wait till job is done and get job result
        mediator_job = wait_for_complete(mediator_job)
        pprint('Extract faces job result: ')
        pprint(mediator_job)

        pprint('face_extraction_id:')
        face_extraction_id = mediator_job.job_output.face_extraction_id
        pprint(face_extraction_id)

        pprint('Querying Face API for extraction info ...')
        api_client = facerecognition_client.ApiClient(mediator_config)
        face_recognition_api = facerecognition_client.FaceRecognitionApi(api_client)
        face_extraction = face_recognition_api.get_face_extraction_collection(project_service_id, face_extraction_id)
        pprint(face_extraction)

        pprint('Querying Face API for first face info ...')
        face = face_recognition_api.get_face(project_service_id, face_extraction.face_ids[0])
        pprint(face)

        pprint('Preparing cluster faces job ...')
        job_mediator_input = create_job_mediator_input_cluster_faces(face_extraction_id)
        pprint(job_mediator_input)

        # Submit the input job to Mediator
        pprint('Submitting cluster faces job ...')
        mediator_job = submit_job(job_mediator_input)
        pprint(job_mediator_input)

        # Wait till job is done and get job result
        mediator_job = wait_for_complete(mediator_job)
        pprint('Cluster faces job result: ')
        pprint(mediator_job)

        pprint('cluster_collection_id:')
        cluster_collection_id = mediator_job.job_output.cluster_collection_id
        pprint(cluster_collection_id)

        pprint('Querying Face API for cluster collection info ...')
        cluster_collection = face_recognition_api.get_cluster_collection(project_service_id, cluster_collection_id)
        pprint(cluster_collection)

        upload_image_to_s3()

        pprint('Receiving signed url ...')
        input_url = get_signed_image_url_input()

        pprint('input_url:')
        pprint(input_url)

        job_mediator_input = create_job_mediator_input_search_faces(input_url, cluster_collection_id)
        pprint(job_mediator_input)

        # Submit the input job to Mediator
        pprint('Submitting search faces job ...')
        mediator_job = submit_job(job_mediator_input)
        pprint(job_mediator_input)

        # Wait till job is done and get job result
        mediator_job = wait_for_complete(mediator_job)
        pprint('Search faces job result: ')
        pprint(mediator_job)

    except ApiException as e:
        print("Exception when calling api: %s\n" % e)


main()
