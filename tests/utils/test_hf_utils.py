# file: circuit-tracer/utils/test_hf_utils.py

import unittest
from unittest import mock

# Import the function and exceptions you need to test or mock
from circuit_tracer.utils.hf_utils import download_hf_uris
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

# A dummy URI for all tests
TEST_URI = "hf://test-org/test-repo/model.bin"


class TestHfUtilsDownload(unittest.TestCase):
    """Test suite for the download_hf_uris function."""

    @mock.patch("circuit_tracer.utils.hf_utils.hf_hub_download")
    @mock.patch("circuit_tracer.utils.hf_utils.hf_api.repo_info")
    @mock.patch("circuit_tracer.utils.hf_utils.get_token", return_value=None)
    def test_public_no_token(self, mock_get_token, mock_repo_info, mock_download):
        """Tests a public repo download with no token."""
        mock_repo_info.return_value = mock.MagicMock(private=False, gated=False)
        mock_download.return_value = "/fake/path/model.bin"
        result = download_hf_uris([TEST_URI])
        self.assertEqual(result, {TEST_URI: "/fake/path/model.bin"})
        mock_download.assert_called_once()

    @mock.patch("circuit_tracer.utils.hf_utils.hf_hub_download")
    @mock.patch("circuit_tracer.utils.hf_utils.hf_api.repo_info")
    @mock.patch("circuit_tracer.utils.hf_utils.get_token", return_value="fake_token")
    def test_public_with_token(self, mock_get_token, mock_repo_info, mock_download):
        """Tests a public repo download with a token present."""
        mock_repo_info.return_value = mock.MagicMock(private=False, gated=False)
        mock_download.return_value = "/fake/path/model.bin"
        result = download_hf_uris([TEST_URI])
        self.assertEqual(result, {TEST_URI: "/fake/path/model.bin"})
        mock_download.assert_called_once()

    @mock.patch("circuit_tracer.utils.hf_utils.hf_hub_download")
    @mock.patch("circuit_tracer.utils.hf_utils.hf_api.repo_info")
    @mock.patch("circuit_tracer.utils.hf_utils.get_token", return_value="fake_token_with_access")
    def test_gated_with_access(self, mock_get_token, mock_repo_info, mock_download):
        """Tests a gated repo where the user has access."""
        mock_repo_info.return_value = mock.MagicMock(private=False, gated=True)
        mock_download.return_value = "/fake/path/model.bin"
        result = download_hf_uris([TEST_URI])
        self.assertEqual(result, {TEST_URI: "/fake/path/model.bin"})
        mock_download.assert_called_once()

    @mock.patch("circuit_tracer.utils.hf_utils.hf_hub_download")
    @mock.patch("circuit_tracer.utils.hf_utils.hf_api.repo_info")
    @mock.patch("circuit_tracer.utils.hf_utils.get_token", return_value="fake_token_no_access")
    def test_gated_without_access(self, mock_get_token, mock_repo_info, mock_download):
        """Tests a gated repo where the user lacks access.
        The download is attempted and raises GatedRepoError.
        """
        # Setup: Pre-flight check passes, as repo_info just returns metadata.
        mock_repo_info.return_value = mock.MagicMock(private=False, gated=True)
        # Setup: The download itself will fail.
        mock_download.side_effect = GatedRepoError("User has not accepted terms.")

        # Execute & Assert: Check that the GatedRepoError is raised by the function.
        with self.assertRaises(GatedRepoError):
            download_hf_uris([TEST_URI])

        # Assert that the download was actually attempted.
        mock_download.assert_called_once()

    @mock.patch("circuit_tracer.utils.hf_utils.hf_hub_download")
    @mock.patch("circuit_tracer.utils.hf_utils.hf_api.repo_info")
    @mock.patch("circuit_tracer.utils.hf_utils.get_token", return_value=None)
    def test_gated_no_token(self, mock_get_token, mock_repo_info, mock_download):
        """Tests a gated repo when no token is available."""
        # Setup: Pre-flight check passes, as repo_info just returns metadata.
        mock_repo_info.return_value = mock.MagicMock(private=False, gated=True)

        with self.assertRaises(PermissionError):
            download_hf_uris([TEST_URI])

        # no need to attempt download as no token means no access to a gated repo
        mock_download.assert_not_called()

    @mock.patch("circuit_tracer.utils.hf_utils.hf_hub_download")
    @mock.patch("circuit_tracer.utils.hf_utils.hf_api.repo_info")
    @mock.patch("circuit_tracer.utils.hf_utils.get_token", return_value="fake_token")
    def test_private_no_access_or_non_existent(self, mock_get_token, mock_repo_info, mock_download):
        """Tests a private repo the user can't see, or a repo that doesn't exist.
        Pre-flight check fails and error is propagated
        """
        mock_repo_info.side_effect = RepositoryNotFoundError("Repo not found.")

        with self.assertRaises(RepositoryNotFoundError):
            download_hf_uris([TEST_URI])

        # download is not called, as the repo is not found
        mock_download.assert_not_called()
