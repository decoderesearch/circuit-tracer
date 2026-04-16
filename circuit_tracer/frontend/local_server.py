import atexit
import functools
import gzip
import http.server
import json
import logging
import os
import socketserver
import threading
from importlib.resources import files
from pathlib import Path

logger = logging.getLogger(__name__)
logger.propagate = False

DEFAULT_FRONTEND_DIR = files("circuit_tracer") / "frontend/assets"


class ListHandler(logging.Handler):
    """Handler that appends log records to a list."""

    def __init__(self, log_list):
        super().__init__()
        self.log_list = log_list

    def emit(self, record):
        msg = self.format(record)
        self.log_list.append(msg)


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


# Create handler for serving circuit graph data
# Local feature handling incorporated
class CircuitGraphHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, frontend_dir, data_dir, features_dir, **kwargs):
        self.data_dir = data_dir
        self.features_dir = features_dir
        super().__init__(*args, directory=str(frontend_dir), **kwargs)

    def log_message(self, format, *args):
        message = format % args
        logger.info(
            "%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), message)
        )

    def do_GET(self):
        try:
            self._do_GET()
        except Exception as e:
            logger.exception(f"Error handling GET request: {e}")
            self.send_response(500)
            self.end_headers()

    def _do_GET(self):
        logger.info(f"Received request for {self.path}")

        if (self.features_dir is not None) and (self.path.startswith("/features/")):
            rel_path = self.path[len("/features/") :].split("?")[0]
            local_path = os.path.join(self.features_dir, rel_path)
            if not os.path.exists(local_path):
                self.send_response(404)
                self.end_headers()
                return
            range_header = self.headers.get("Range", "")
            with open(local_path, "rb") as f:
                if range_header.startswith("bytes="):
                    file_size = os.path.getsize(local_path)
                    start, end = range_header[6:].split("-")
                    start = int(start)
                    end = int(end) if end else file_size - 1
                    f.seek(start)
                    content = f.read(end - start + 1)
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                else:
                    content = f.read()
                    self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
            return

        # Handle data and graph_data requests from local storage
        if self.path.startswith(("/data/", "/graph_data/")):
            # Extract the file path from the URL
            if self.path.startswith("/data/"):
                rel_path = self.path[len("/data/") :].split("?")[0]
            else:  # /graph_data/
                rel_path = self.path[len("/graph_data/") :].split("?")[0]

            # Properly join paths to handle missing slashes
            local_path = os.path.join(self.data_dir, rel_path)

            logger.info(
                f"Rewritten path to {local_path}. "
                f"(self.path: {self.path}; self.data_dir: {self.data_dir})"
            )
            if not os.path.exists(local_path):
                self.send_response(404)
                self.end_headers()
                return

            self.send_response(200)
            with open(local_path, "rb") as f:
                content = f.read()

            # Compress large responses
            if len(content) > 1024**2:  # 1MB threshold
                content = gzip.compress(content, compresslevel=3)
                self.send_header("Content-Encoding", "gzip")

            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/save_graph/"):
            self.send_response(404)
            return

        try:
            # Extract scan and slug from the URL path
            parts = self.path.split("?")[0].strip("/").split("/")
            slug = parts[-1]

            logger.info(f"Saving graph for {slug}")

            # Read the request body
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))

            # Generate filename with timestamp
            save_path = os.path.join(self.data_dir, f"{slug}.json")

            # Read the existing file and update it
            with open(save_path) as f:
                graph = json.load(f)
                graph["qParams"] = data["qParams"]

            with open(save_path, "w") as f:
                json.dump(graph, f, indent=2)

            self.send_response(200)
            self.end_headers()
            logger.info(f"Graph saved: {save_path}")

        except Exception as e:
            logger.exception(f"Error saving graph: {e}")
            self.send_response(500)
            self.end_headers()


class Server:
    def __init__(self, httpd, server_thread):
        self.httpd = httpd
        self.server_thread = server_thread
        self.logs = []
        self._stopped = False  # Initialize the flag here

        # Add a handler to logger that records to self.logs
        self.log_handler = ListHandler(self.logs)
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(self.log_handler)
        logger.setLevel(logging.INFO)
        # Register shutdown with atexit
        atexit.register(self.stop)

    def stop(self):
        # Check if already stopped to prevent multiple calls
        if self._stopped:
            return
        self._stopped = True

        logger.info("Stopping server...")

        try:
            # First, stop accepting new connections
            self.httpd.socket.close()
        except Exception as e:
            logger.debug(f"Error closing socket: {e}")

        # Then shutdown the server
        shutdown_thread = threading.Thread(target=self.httpd.shutdown)
        shutdown_thread.daemon = True
        shutdown_thread.start()

        # Wait with timeout for threads to complete
        shutdown_thread.join(timeout=5)
        self.server_thread.join(timeout=5)

        # Force socket close regardless of shutdown success
        try:
            self.httpd.server_close()
        except Exception as e:
            logger.debug(f"Error during server_close: {e}")

        logger.info("Server stopped")

        # Remove our handler when the server stops
        logger.removeHandler(self.log_handler)

        # Unregister from atexit to avoid duplicate calls
        atexit.unregister(self.stop)

    def get_logs(self):
        """Return the current log messages."""
        return self.logs


def serve(data_dir, frontend_dir=None, features_dir=None, port=8032):
    """Start a local HTTP server in a separate thread.

    Args:
        data_dir: Directory for local graph data.
        frontend_dir: Directory containing frontend files. Defaults to DEFAULT_FRONTEND_DIR.
        features_dir: If features are local, path to directory containing local feature files; else None. Defaults to None.
        port: Port to serve on. Defaults to 8032.

    Returns:
        Server object with a stop() method to shut down the server.
    """

    # Use provided directories or defaults
    frontend_dir = Path(frontend_dir).resolve() if frontend_dir else DEFAULT_FRONTEND_DIR

    frontend_dir_path = Path(frontend_dir)  # type: ignore
    if not frontend_dir_path.exists() and frontend_dir_path.is_dir():
        raise ValueError(f"Got frontend dir {frontend_dir} but this is not a valid directory")

    logger.info(f"Serving files from: {frontend_dir}")

    # Create a partially applied handler class with configured directories
    handler = functools.partial(
        CircuitGraphHandler, frontend_dir=frontend_dir, data_dir=data_dir, features_dir=features_dir
    )

    httpd = ReusableTCPServer(("", port), handler)

    # Start the server in a thread
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    logger.info(f"Serving at http://localhost:{port}")
    logger.info(f"Serving files from: {frontend_dir}")
    logger.info(f"Serving data from: {data_dir}")

    return Server(httpd, server_thread)
