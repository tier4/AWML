#! /usr/bin/env python

import argparse
import json
from collections import OrderedDict
from typing import Any, Dict, List
from urllib.parse import urljoin

from attrs import define
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString
from webautoauth import requests
from webautoauth.token import HttpService, TokenSource, load_config


@define
class DatasetMetadata:
    """Represents an metadata information of each dataset.

    Attributes:
        id (str): id in Web.Auto for each dataset. e.g, be882f53-0e90-438a-8da3-caed58d4348e
        name (str): name in Web.Auto for each dataset. e.g, DBv0.0_tokyo_prd-example_bc143e44-08d4-4f65-a95e-12fc7de85ea3_2024-08-30_09-43-48_09-44-08
    """

    id: str
    name: str


class AnnotationDatasetSearchClient:
    """
    A client for searching annotation datasets using the Web.Auto API.

    This client handles authentication and pagination automatically,
    allowing you to search for annotation datasets based on keywords
    and project IDs.
    """

    def __init__(self):
        """
        Initializes the AnnotationDatasetSearchClient.
        """
        config = load_config()
        token_source = TokenSource(HttpService(config))
        self._session = requests.make_session(token_source)

        self._page_size = 100
        self._base_url = urljoin("https://api.data-search.tier4.jp", "/v1/projects/annotation_datasets/search")
        self._headers = {"content-type": "application/json"}

    def search(
        self,
        keyword: str,
        project_ids: List[str] = ["prd_jt", "x2_dev"],
        approved: bool = True,
    ) -> List[DatasetMetadata]:
        """
        Searches for annotation datasets based on the provided keyword and project IDs.

        Args:
            keyword (str): The keyword to search for in the dataset names.
            project_ids (List[str]): A list of project IDs to filter the search results. Default is ["prd_jt", "x2_dev"].
            approved (bool): Whether to filter by approved datasets. Default is True.

        Returns:
            List[DatasetMetadata]: List containing dataset IDs and names that match the search criteria.
                Returns empty lists if no matches are found.
        """
        payload = {
            "approved": approved,
            "deprecated": False,
            "name_keyword": keyword,
            "project_ids": project_ids,
        }
        results = self._fetch_all_annotation_datasets(payload)
        return [DatasetMetadata(id=data["id"], name=data["name"]) for data in results]

    def _fetch_all_annotation_datasets(self, payload: Dict[str, Any]):
        """
        Fetches all annotation datasets matching the provided payload, handling pagination.

        Args:
            payload (Dict[str, Any]): The search criteria payload to send in the POST request.

        Returns:
            List[Dict[str, Any]]: A list of all dataset items retrieved from the API.
        """
        all_items = []
        next_page_token = None

        while True:
            params = {"page_size": self._page_size}
            if next_page_token:
                params["page_token"] = next_page_token

            response = self._session.post(
                url=self._base_url, headers=self._headers, params=params, data=json.dumps(payload)
            )
            data = response.json()

            all_items.extend(data["items"])
            next_page_token = data.get("next_page_token")

            if not next_page_token:
                break

        return all_items


def search_dataset_metadata(keyword: str, project_ids: List[str]) -> List[DatasetMetadata]:
    """
    Search for dataset metadata informations using the provided keyword and project IDs.

    This function creates an AnnotationDatasetSearchClient instance and uses it to search
    for datasets that match the given keyword within the specified projects. It prints
    the number of datasets found and returns both their IDs and names.

    Args:
        keyword (str): The search keyword to filter datasets. This will be matched against
            dataset names in the search.
        project_ids (List[str]): List of project identifiers to search within. Only datasets
            belonging to these projects will be returned.

    Returns:
        List[DatasetMetadata]: List containing dataset IDs and names that match the search criteria.
            Returns empty lists if no matches are found.

    Example:
        >>> metadata = search_dataset_metadata("DBv1.3", ["prd_jt", "x2_dev"])
        2 found for keyword: DBv1.3 within projects: ['prd_jt', 'x2_dev']
        >>> print(metadata[0].id)
        'xxx'
        >>> print((metadata[0].name)
        'DBv0.0_tokyo_car1_1970-01-01'
    """
    client = AnnotationDatasetSearchClient()
    found_dataset_metadata = client.search(keyword=keyword, project_ids=project_ids)
    print(f"{len(found_dataset_metadata)} found for keyword: {keyword} within projects: {project_ids}")
    return found_dataset_metadata


def update_database_config(config_file_path: str, found_dataset_metadata: List[DatasetMetadata]) -> None:
    """
    Update the database configuration file with new datasets and remove deprecated ones.

    This function reads a YAML configuration file, updates it with newly found datasets,
    and removes any datasets that are no longer found in the search results.

    Args:
        config_file_path (str): Path to the YAML configuration file that needs to be updated.
            The file should contain 'train', 'val', and 'test' lists of dataset IDs.
        found_dataset_metadata (List[DatasetMetadata]): List containing dataset IDs and names that match the search criteria.
            Any IDs in the config file that are not in this list will be removed.

    Note:
        The function maintains the YAML file format with dataset names as comments next to their IDs.
        For example:
        train:
          - dataset_id_1  # Dataset Name 1
          - dataset_id_2  # Dataset Name 2
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(config_file_path, "r") as f:
        config_content = yaml.load(f)

    is_config_content_updated = False

    # If the found ID does not exist in the config file, add it.
    for found_dataset_id, found_dataset_name in ((d.id, d.name) for d in found_dataset_metadata):
        if found_dataset_id not in config_content["train"] + config_content["val"] + config_content["test"]:
            print(f"{found_dataset_id} seems to be a new ID, adding to the config file")
            config_content["train"].append(found_dataset_id)
            config_content["train"].yaml_add_eol_comment(found_dataset_name, len(config_content["train"]) - 1)
            is_config_content_updated = True

    # If the ID from config file is not found, remove it. (probably deprecated)
    found_dataset_ids: List[str] = [d.id for d in found_dataset_metadata]
    for split_name in ["train", "val", "test"]:
        dataset_ids = config_content[split_name].copy()
        for dataset_id_from_config_file in dataset_ids:
            if dataset_id_from_config_file not in found_dataset_ids:
                print(f"{dataset_id_from_config_file} not found in search results, removing from {split_name} split")
                config_content[split_name].remove(dataset_id_from_config_file)
                is_config_content_updated = True

    # Save the results if the content is updated
    if is_config_content_updated:
        # Ensure 'docs' field is treated as a literal block
        if "docs" in config_content:
            config_content["docs"] = LiteralScalarString(config_content["docs"])

        with open(config_file_path, "w") as f:
            yaml.dump(config_content, f)
        print("Config file updated with new datasets.")
    else:
        print("No new datasets found.")


def main():
    parser = argparse.ArgumentParser(description="Search for Annotation Datasets")
    parser.add_argument("--config_path", type=str, required=True, help="Config path to update")
    parser.add_argument("--keyword", type=str, required=True, help="The keyword to search for")
    parser.add_argument(
        "--project_ids",
        type=str,
        nargs="*",
        default=["prd_jt", "x2_dev"],
        help="List of project IDs to filter by (optional)",
    )
    args = parser.parse_args()
    assert len(args.project_ids) > 0, "You MUST provide at least one project id."

    found_dataset_metadata = search_dataset_metadata(args.keyword, args.project_ids)
    update_database_config(args.config_path, found_dataset_metadata)


if __name__ == "__main__":
    main()
