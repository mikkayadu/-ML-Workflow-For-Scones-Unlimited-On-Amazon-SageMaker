
#First Lambda function, Serialize_Image_Data
import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']
    
    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, '/tmp/image.png')
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')  # Decoding to utf-8 to convert bytes to string

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }



# SECOND LAMBDA FUNCTION, DECODER
import json
import sagemaker
import base64
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2023-10-21-17-56-00-230"

def lambda_handler(event, context):

    # Decode the image data from the event's body
    image = base64.b64decode(event['body']['image_data'])

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(endpoint_name=ENDPOINT)

    # For this model, the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    inferences = predictor.predict(image)
    
    # We return the data back to the Step Function    
    event['inferences'] = inferences.decode('utf-8')
    print(event)
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }



#THIRD LAMBDA FUNCTION , THRESHOLD_CLASSIFIER
import json

# Define the threshold
THRESHOLD = 0.7

def lambda_handler(event, context):
    
    event_body = json.loads(event['body'])
    print(event_body)
    
    inferences_str = event_body['inferences']
    print(inferences_str)
    
    inferences = [float(num) for num in inferences_str.strip("[]").split(", ")]
    print("Inference", inferences)
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(inference > THRESHOLD for inference in inferences)
    print("meets_threshold", meets_threshold)
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if not meets_threshold:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event_body)
    }

