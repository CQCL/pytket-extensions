# Copyright 2020-2022 Cambridge Quantum Computing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable, Optional
from abc import ABC, abstractmethod
from itertools import takewhile, count
import keyring


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


class CredentialStorage(ABC):
    """Abstract class for managing credentials and tokens for the Honeywell API."""

    @abstractmethod
    def save_login_credential(self, user_name: str, password: str) -> None:
        """Save the user_name and password"""
        ...

    @abstractmethod
    def login_credential(self, user_name: str) -> Optional[str]:
        """Retrieve the password associated with the user_name"""
        ...

    @abstractmethod
    def save_tokens(self, id_token: str, refresh_token: str) -> None:
        """Save the id_token and refresh_token"""
        ...

    @abstractmethod
    def save_refresh_token(self, refresh_token: str) -> None:
        """Save the refresh_token"""
        ...

    @property
    @abstractmethod
    def id_token(self) -> Optional[str]:
        """Retrieve the id_token"""
        ...

    @property
    @abstractmethod
    def refresh_token(self) -> Optional[str]:
        """Retrieve the refresh_token"""
        ...

    @abstractmethod
    def delete_login_credential(self, user_name: str) -> None:
        """Delete the stored username and password"""
        ...

    @abstractmethod
    def delete_tokens(self) -> None:
        """Delete the refresh_token and id_token"""
        ...


class MemoryStorage(CredentialStorage):

    """In memory credential storage"""

    def __init__(self) -> None:
        self._user_name: Optional[str] = None
        self._password: Optional[str] = None
        self._id_token: Optional[str] = None
        self._refresh_token: Optional[str] = None

    def save_login_credential(self, user_name: str, password: str) -> None:
        self._user_name = user_name
        self._password = password

    def login_credential(self, user_name: str) -> Optional[str]:
        self._user_name = user_name
        return None if self._password is None else self._password

    def save_tokens(self, id_token: str, refresh_token: str) -> None:
        self._id_token = id_token
        self._refresh_token = refresh_token

    def save_refresh_token(self, refresh_token: str) -> None:
        self._refresh_token = refresh_token

    @property
    def id_token(self) -> Optional[str]:
        return self._id_token

    @property
    def refresh_token(self) -> Optional[str]:
        return self._refresh_token

    def delete_login_credential(self, user_name: str) -> None:
        self._user_name = None
        self._password = None

    def delete_tokens(self) -> None:
        self._id_token = None
        self._refresh_token = None


class PersistentStorage(CredentialStorage):

    """Persistent credential storage using keyring"""

    KEYRING_SERVICE = "HQS-API"

    def save_login_credential(self, user_name: str, password: str) -> None:
        keyring.set_password(self.KEYRING_SERVICE, user_name, password)

    def login_credential(self, user_name: str) -> Optional[str]:
        password = keyring.get_password(self.KEYRING_SERVICE, user_name)
        return None if password is None else password

    def save_tokens(self, id_token: str, refresh_token: str) -> None:
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
                    self.KEYRING_SERVICE, f"{token_name}_{index}", part
                )

    def save_refresh_token(self, refresh_token: str) -> None:
        split_refresh_tokens = list(split_utf8(refresh_token, 512))

        for index, token_part in enumerate(split_refresh_tokens):
            keyring.set_password(  # type: ignore
                self.KEYRING_SERVICE, f"refresh_token_{index}", token_part
            )

    @property
    def id_token(self) -> Optional[str]:
        token = "".join(self._get_token_parts("id_token"))
        return token if token else None

    @property
    def refresh_token(self) -> Optional[str]:
        token = "".join(self._get_token_parts("refresh_token"))
        return token if token else None

    def delete_login_credential(self, user_name: str) -> None:
        try:
            keyring.delete_password(self.KEYRING_SERVICE, user_name)
        except keyring.errors.PasswordDeleteError:
            pass

    def delete_tokens(self) -> None:
        """Delete tokens in the keyring"""
        for token_name in ("id_token", "refresh_token"):
            for index in count():
                token_part = f"{token_name}_{index}"
                try:
                    keyring.delete_password(self.KEYRING_SERVICE, token_part)
                except keyring.errors.PasswordDeleteError:
                    # stop when the token part doesn't exist
                    break

    def _get_token_parts(self, token_name: str) -> Iterable[str]:
        token_parts = (
            keyring.get_password(self.KEYRING_SERVICE, f"{token_name}_{i}")  # type: ignore
            for i in count(start=0)
        )
        return takewhile(lambda x: x is not None, token_parts)  # type: ignore
