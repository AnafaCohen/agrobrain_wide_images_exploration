FROM taranisag/ubuntu20.04-python3.8-datascience

# Create app directory
RUN mkdir -p /app
WORKDIR /app

# Install python requirements
COPY requirements.txt /app
RUN pip3 install -r requirements.txt

#copy app
COPY . /app
