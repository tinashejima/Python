# Lab Monitoring Dashboard

A Flask-based web application that provides a dashboard to monitor lab requests from a Hive database. It allows users to view statistics about different facilities, including the number of orders, total orders, and the last request date.

## Features

-   Dashboard to visualize lab request data.
-   Filter data by facility and date range.
-   View active facilities and their statistics.
-   View the status of orders for each facility.
-   RESTful API for fetching data.

## Technologies Used

-   **Backend**: Flask, Python
-   **Database**: Hive (connected via PyHive)
-   **Data Manipulation**: Pandas
-   **WSGI Server**: Gunicorn
-   **Containerization**: Docker, Docker Compose

## Prerequisites

-   Python 3.9 or higher
-   Docker and Docker Compose
-   Access to a Hive database

## Installation and Setup

**1. Clone the repository:**

```bash
git clone <repository-url>
cd <repository-name>
```

**2. Create a virtual environment and install dependencies:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Configure environment variables:**

Create a `.env` file in the root directory and add the following variables:

```
HIVE_HOST=<your-hive-host>
HIVE_PORT=<your-hive-port>
HIVE_USER=<your-hive-username>
HIVE_PASSWORD=<your-hive-password>
HIVE_DATABASE=<your-hive-database>
FLASK_APP=app.py
FLASK_ENV=development
```

## Usage

### Running Locally

```bash
flask run
```

The application will be available at `http://localhost:5000`.

### Running with Docker

The application is containerized using Docker and can be run using Docker Compose.

**1. Build and run the container:**

```bash
docker-compose -f lims.yml up --build
```

The application will be available at `http://localhost:5000`.

The `lims.yml` file defines the service and its configuration. It uses the `Dockerfile` to build the image.

## API Endpoints

-   `GET /`: Renders the main HTML page for the dashboard.
-   `POST /active_facilities`: Returns a JSON object with a list of active facilities and their statistics.
    -   **Request Body**:
        ```json
        {
            "date": "YYYY-MM-DD",
            "facilities": ["facility_id_1", "facility_id_2"]
        }
        ```
-   `POST /facility_statuses`: Returns a JSON object with the status counts for each facility in a given date range.
    -   **Request Body**:
        ```json
        {
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD",
            "facilities": ["facility_id_1", "facility_id_2"]
        }
        ```

## Dockerization

The `Dockerfile` creates a production-ready image for the application.

-   It installs all necessary system and Python dependencies.
-   It runs the application using `gunicorn` as the WSGI server.
-   A non-root user is created for better security.
-   A health check is included to monitor the application's status.

The `lims.yml` file is used to manage the container's lifecycle, configuration, and networking.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

