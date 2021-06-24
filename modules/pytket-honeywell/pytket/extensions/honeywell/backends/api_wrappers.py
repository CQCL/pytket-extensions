# -*- coding: utf-8 -*-
""""
Functions used to submit jobs with Honeywell Quantum Solutions API.

Adapted from original file provided by Honeywell Quantum Solutions
"""

import datetime
import time
from http import HTTPStatus
from typing import Iterable, Optional, Dict, Any, Tuple
from itertools import takewhile, count
import asyncio
import json
import getpass
import jwt
import requests
from websockets import connect, exceptions  # type: ignore
import nest_asyncio  # type: ignore
import keyring  # type: ignore

from .config import HoneywellConfig

# This is necessary for use in Jupyter notebooks to allow for nested asyncio loops
nest_asyncio.apply()


class HQSAPIError(Exception):
    pass


def split_utf8(s: str, n: int) -> Iterable[str]:
    # stolen from
    # https://stackoverflow.com/questions/6043463/split-unicode-string-into-300-byte-chunks-without-destroying-characters
    """Split UTF-8 s into chunks of maximum length n."""
    s_bytes = s.encode("utf-8")
    while len(s_bytes) > n:
        k = n
        while (s_bytes[k] & 0xC0) == 0x80:
            k -= 1
        yield s_bytes[:k].decode("utf-8")
        s_bytes = s_bytes[k:]
    yield s_bytes.decode("utf-8")


class _OverrideManger:
    def __init__(
        self,
        api_handler: "HoneywellQAPI",
        timeout: Optional[int] = None,
        retry_timeout: Optional[int] = None,
    ):
        self._timeout = timeout
        self._retry = retry_timeout
        self.api_handler = api_handler
        self._orig_timeout = api_handler.timeout
        self._orig_retry = api_handler.retry_timeout

    def __enter__(self) -> None:
        if self._timeout is not None:
            self.api_handler.timeout = self._timeout
        if self._retry is not None:
            self.api_handler.retry_timeout = self._retry

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        self.api_handler.timeout = self._orig_timeout
        self.api_handler.retry_timeout = self._orig_retry


class HoneywellQAPI:

    JOB_DONE = ["failed", "completed", "canceled"]

    DEFAULT_API_URL = "https://qapi.honeywell.com/"
    DEFAULT_TIME_SAFETY = (
        60  # Default safety factor (in seconds) to token expiration before a refresh
    )

    KEYRING_NAME = "HQS-API"

    def __init__(
        self,
        user_name: Optional[str] = None,
        token: Optional[str] = None,
        machine: Optional[str] = None,
        api_url: Optional[str] = None,
        api_version: int = 1,
        use_websocket: bool = True,
        time_safety: Optional[int] = None,
        login: bool = True,
    ):
        """Initialize and login to the Honeywell Quantum API interface

        All arguments are optional

        Arguments:
            user_name (str): User e-mail used to register
            token (str): Token used to refresh id token
            url (str): Url of the Quantum API including version:
             https://qapi.honeywell.com/v1/
            shots (int): Default number of shots for submitted experiments
            use_websockets: Whether to default to using websockets to reduce traffic
        """
        self.config = HoneywellConfig.from_default_config_file()

        self.url = (
            f"{api_url}v{api_version}/"
            if api_url
            else f"{self.DEFAULT_API_URL}v{api_version}/"
        )
        self.keyring_service = self.KEYRING_NAME

        self.user_name = user_name if user_name else self.config.username
        self.refresh_token = token if token else self._get_token("refresh_token")
        self.api_version = api_version
        self.machine = machine
        self.use_websocket = use_websocket
        self.time_safety_factor = (
            time_safety if time_safety else self.DEFAULT_TIME_SAFETY
        )
        self.ws_timeout = 180
        self.retry_timeout = 5
        self.timeout: Optional[int] = None  # don't timetout by default

        if login:
            self.login()

    def override_timeouts(
        self, timeout: Optional[int] = None, retry_timeout: Optional[int] = None
    ) -> _OverrideManger:
        return _OverrideManger(self, timeout=timeout, retry_timeout=retry_timeout)

    def _request_tokens(self, body: dict) -> Tuple[Optional[int], Optional[Any]]:
        """Method to send login request to machine api and save tokens."""
        try:
            # send request to login
            response = requests.post(
                f"{self.url}/login",
                json.dumps(body),
            )

            # reset body to delete credentials
            body = {}

            if response.status_code != HTTPStatus.OK:
                return response.status_code, response.json()

            else:
                print("***Successfully logged in***")
                self._save_tokens(
                    response.json()["id-token"], response.json()["refresh-token"]
                )
                return response.status_code, None

        except requests.exceptions.RequestException as e:
            print(e)
            return None, None

    def _get_credentials(self) -> Tuple[str, str]:
        """Method to ask for user's credentials"""
        if self.config.username is not None:
            pwd = keyring.get_password(  # type: ignore
                self.keyring_service,
                self.config.username,
            )
            if pwd:
                self.user_name = self.config.username
                return self.user_name, pwd

        if not self.user_name:
            user_name = input("Enter your email: ")
            self.user_name = user_name

        pwd = getpass.getpass(prompt="Enter your password: ")
        return self.user_name, pwd

    def _authenticate(
        self,
        action: Optional[str] = None,
        __user_name: Optional[str] = None,
        __pwd: Optional[str] = None,
    ) -> None:
        """This method makes requests to refresh or get new id-token.
        If a token refresh fails due to token being expired, credentials
        get requested from user.

        The __user_name and __pwd parameters are there for debugging
        purposes only, do not include your credentials in source code.
        """
        # login body
        body = {}

        if action == "refresh":
            body["refresh-token"] = self.refresh_token
        else:
            # ask user for crendentials before making login request
            user_name: Optional[str]
            pwd: Optional[str]
            if __user_name and __pwd:
                user_name = __user_name
                pwd = __pwd
            else:
                user_name, pwd = self._get_credentials()
            body["email"] = user_name
            body["password"] = pwd

            # clear credentials
            user_name = None
            pwd = None

        # send login request to API
        status_code, message = self._request_tokens(body)

        body = {}

        if status_code != HTTPStatus.OK:
            # check if we got an error because refresh token has expired
            if status_code == HTTPStatus.BAD_REQUEST:
                if message is not None:
                    if "Invalid Refresh Token" in message["error"]["text"]:
                        # ask user for credentials to login again
                        user_name, pwd = self._get_credentials()
                        body["email"] = user_name
                        body["password"] = pwd

                        # send login request to API
                        status_code, message = self._request_tokens(body)
                else:
                    raise HQSAPIError("No message with BAD_REQUEST")

        if status_code != HTTPStatus.OK:
            raise HQSAPIError(
                "HTTP error while logging in: "
                f"{status_code} {'' if message is None else message}"
            )

    def _get_token_parts(self, token_name: str) -> Iterable[str]:
        token_parts = (
            keyring.get_password(self.keyring_service, f"{token_name}_{i}")  # type: ignore
            for i in count(start=0)
        )
        return takewhile(lambda x: x is not None, token_parts)  # type: ignore

    def _get_token(self, token_name: str):  # type: ignore
        """Method to retrieve id and refresh tokens from system's keyring service.
        Windows keyring backend has a length limitation on passwords.
        To avoid this, passwords get split in to tokens of length 512.
        """

        token = "".join(self._get_token_parts(token_name))
        return token if token else None

    def _save_tokens(self, id_token: str, refresh_token: str) -> None:
        """Method to save id and refresh tokens on system's keyring service.
        Windows keyring backend has a length limitation on passwords.
        To avoid this, passwords get split in to tokens of length 512.
        """

        split_id_tokens = list(split_utf8(id_token, 512))
        split_refresh_tokens = list(split_utf8(refresh_token, 512))

        for token_list, token_name in zip(
            (split_id_tokens, split_refresh_tokens), ("id_token", "refresh_token")
        ):
            for index, part in enumerate(token_list):
                keyring.set_password(  # type: ignore
                    self.keyring_service, f"{token_name}_{index}", part
                )

    def login(self) -> str:
        """This methods checks if we have a valid (non-expired) id-token
        and returns it, otherwise it gets a new one with refresh-token.
        If refresh-token doesn't exist, it asks user for credentials.
        """
        # check if id_token exists
        id_token = self._get_token("id_token")
        if id_token is None:
            # authenticate against '/login' endpoint
            self._authenticate()

            # get id_token
            id_token = self._get_token("id_token")
        if id_token is None:
            raise HQSAPIError("Unable to retrieve id token.")
        # check id_token is not expired yet
        expiration_date = jwt.decode(id_token, verify=False)["exp"]
        if expiration_date < (
            datetime.datetime.now().timestamp() - self.time_safety_factor
        ):
            print("Your id token is expired. Refreshing...")

            # get refresh_token
            refresh_token = self._get_token("refresh_token")
            if refresh_token is not None:
                self._authenticate("refresh")
            else:
                self._authenticate()

            # get id_token
            id_token = self._get_token("id_token")

        return id_token  # type: ignore

    def delete_authentication(self) -> None:
        """Use keyrings delete_password to remove stored"""
        for token_name in ("id_token", "refresh_token"):
            for index in count():
                token_part = f"{token_name}_{index}"
                try:
                    keyring.delete_password(self.keyring_service, token_part)  # type: ignore
                except keyring.errors.PasswordDeleteError:
                    # stop when the token part doesn't exist
                    break

    def recent_jobs(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        days: Optional[int] = None,
        jobs: Optional[int] = None,
    ) -> Any:
        id_token = self.login()
        if start is not None and end is not None:
            res = requests.get(
                f"{self.url}metering?start={start}&end={end}",
                headers={"Authorization": id_token},
            )
            self._response_check(res, f"metering between {start} and {end}")
            return res.json()
        elif days is not None:
            res = requests.get(
                f"{self.url}metering?days={days}", headers={"Authorization": id_token}
            )
            self._response_check(res, f"metering of last {days} days")
            return res.json()
        elif jobs is not None:
            res = requests.get(
                f"{self.url}metering?jobs={jobs}", headers={"Authorization": id_token}
            )
            self._response_check(res, f"metering of last {jobs} jobs")
            return res.json()
        else:
            raise ValueError("Need more information to make a metering request")

    def submit_job(
        self,
        qasm_str: str,
        shots: Optional[int] = None,
        machine: Optional[str] = None,
        name: str = "job",
    ) -> str:
        """
        Submits job to device and returns job ID.

        Args:
            qasm_str:   OpenQASM file to run
            shots:      number of repetitions of qasm_str
            machine:    machine to run on
            name:       name of job (for error handling)

        Returns:
            (str):     id of job submitted

        """
        try:
            if not machine and not self.machine:
                raise ValueError("Must provide valid machine name")
            # send job request
            body = {
                "machine": machine if machine else self.machine,
                "name": name,
                "language": "OPENQASM 2.0",
                "program": qasm_str,
                "priority": "normal",
                "count": shots,
                "options": None,
            }
            id_token = self.login()
            res = requests.post(
                f"{self.url}job", json.dumps(body), headers={"Authorization": id_token}
            )
            self._response_check(res, "job submission")

            # extract job ID from response
            jr = res.json()
            job_id: str = str(jr["job"])
            print(
                f"submitted {name} id={{job}}, submit date={{submit-date}}".format(**jr)
            )

        except ConnectionError as e:
            raise e

        return job_id

    def _response_check(self, res: requests.Response, description: str) -> None:
        """Consolidate as much error-checking of response"""
        # check if token has expired or is generally unauthorized
        if res.status_code == HTTPStatus.UNAUTHORIZED:
            jr = res.json()
            raise HQSAPIError(
                (
                    f"Authorization failure attempting: {description}."
                    "\n\nServer Response: {jr}"
                )
            )
        elif res.status_code != HTTPStatus.OK:
            jr = res.json()
            raise HQSAPIError(
                f"HTTP error attempting: {description}.\n\nServer Response: {jr}"
            )

    def retrieve_job_status(
        self, job_id: str, use_websocket: Optional[bool] = None
    ) -> Optional[Dict]:
        """
        Retrieves job status from device.

        Args:
            job_id:        unique id of job
            use_websocket: use websocket to minimize interaction

        Returns:
            (dict):        output from API

        """
        job_url = f"{self.url}job/{job_id}"
        # Using the login wrapper we will automatically try to refresh token
        id_token = self.login()
        if use_websocket or (use_websocket is None and self.use_websocket):
            job_url += "?websocket=true"
        res = requests.get(job_url, headers={"Authorization": id_token})

        jr: Optional[Dict] = None
        # Check for invalid responses, and raise an exception if so
        self._response_check(res, "job status")
        # if we successfully got status return the decoded details
        if res.status_code == HTTPStatus.OK:
            jr = res.json()
        return jr

    def retrieve_job(
        self, job_id: str, use_websocket: Optional[bool] = None
    ) -> Optional[Dict]:
        """
        Retrieves job from device.

        Args:
            job_id:        unique id of job
            use_websocket: use websocket to minimize interaction

        Returns:
            (dict):        output from API

        """
        jr = self.retrieve_job_status(job_id, use_websocket)
        if not jr:
            raise HQSAPIError(f"Unable to retrive job {job_id}")
        if "status" in jr and jr["status"] in self.JOB_DONE:
            return jr

        if "websocket" in jr:
            # wait for job completion using websocket
            jr = asyncio.get_event_loop().run_until_complete(self._wait_results(job_id))
        else:
            # poll for job completion
            jr = self._poll_results(job_id)
        return jr

    def _poll_results(self, job_id: str) -> Optional[Dict]:
        jr = None
        start_time = time.time()
        while True:
            if self.timeout is not None and time.time() > (start_time + self.timeout):
                break
            self.login()
            try:
                jr = self.retrieve_job_status(job_id)

                # If we are failing to retrieve status of any kind, then fail out.
                if jr is None:
                    break
                if "status" in jr and jr["status"] in self.JOB_DONE:
                    return jr
                time.sleep(self.retry_timeout)
            except KeyboardInterrupt:
                raise RuntimeError("Keyboard Interrupted")
        return jr

    async def _wait_results(self, job_id: str) -> Optional[Dict]:
        start_time = time.time()
        while True:
            if self.timeout is not None and time.time() > (start_time + self.timeout):
                break
            self.login()
            jr = self.retrieve_job_status(job_id, True)
            if jr is None:
                return jr
            elif "status" in jr and jr["status"] in self.JOB_DONE:
                return jr
            else:
                task_token = jr["websocket"]["task_token"]
                execution_arn = jr["websocket"]["executionArn"]
                websocket_uri = self.url.replace("https://", "wss://ws.")
                async with connect(websocket_uri) as websocket:
                    body = {
                        "action": "OpenConnection",
                        "task_token": task_token,
                        "executionArn": execution_arn,
                    }
                    await websocket.send(json.dumps(body))
                    while True:
                        try:
                            res = await asyncio.wait_for(
                                websocket.recv(), timeout=self.ws_timeout
                            )
                            jr = json.loads(res)
                            if not isinstance(jr, Dict):
                                raise RuntimeError("Unable to decode response.")
                            if "status" in jr and jr["status"] in self.JOB_DONE:
                                return jr
                        except (
                            asyncio.TimeoutError,
                            exceptions.ConnectionClosed,
                        ):
                            try:
                                # Try to keep the connection alive...
                                pong = await websocket.ping()
                                await asyncio.wait_for(pong, timeout=10)
                                continue
                            except asyncio.TimeoutError:
                                # If we are failing, wait a little while,
                                #  then start from the top
                                await asyncio.sleep(self.retry_timeout)
                                break
                        except KeyboardInterrupt:
                            raise RuntimeError("Keyboard Interrupted")

    def run_job(
        self, qasm_str: str, shots: int, machine: str, name: str = "job"
    ) -> Optional[Dict]:
        """
        Submits a job and waits to receives job result dictionary.

        Args:
            qasm_file:  OpenQASM file to run
            name:       name of job (for error handling)
            shots:      number of repetitions of qasm_str
            machine:    machine to run on

        Returns:
            jr:         (dict) output from API

        """
        job_id = self.submit_job(
            qasm_str=qasm_str, shots=shots, machine=machine, name=name
        )

        jr = self.retrieve_job(job_id)

        return jr

    def status(self, machine: Optional[str] = None) -> str:
        """
        Check status of machine.

        Args:
            (str):    machine name

        """
        id_token = self.login()
        res = requests.get(
            f"{self.url}machine/{machine if machine else self.machine}",
            headers={"Authorization": id_token},
        )
        self._response_check(res, "get machine status")
        jr = res.json()

        return str(jr["state"])

    def cancel(self, job_id: str) -> dict:
        """
        Cancels job.

        Args:
            job_id:     job ID to cancel

        Returns:
            jr:         (dict) output from API

        """

        id_token = self.login()
        res = requests.post(
            f"{self.url}job/{job_id}/cancel", headers={"Authorization": id_token}
        )
        self._response_check(res, "job cancel")
        jr = res.json()

        return jr  # type: ignore
