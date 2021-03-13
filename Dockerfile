# Image to run the application on
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:21.02-py3
FROM ${BASE_IMAGE}


#
# install pre-requisite packages
#
RUN apt-get update && \
    apt-get install -y git


# Install the dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Mount all the application files
COPY . . 

# Run the test
CMD ["python", "-m", "unittest", "tests/test_example.py"]