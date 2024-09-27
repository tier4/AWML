#! /usr/bin/env python

import json
import argparse
from collections import OrderedDict
from typing import Any, Dict, List
from urllib.parse import urljoin

from webautoauth import requests
from webautoauth.token import load_config, TokenSource, HttpService
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString


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
        self._base_url = urljoin("https://api.data-search.tier4.jp",
                                 "/v1/projects/annotation_datasets/search")
        self._headers = {'content-type': 'application/json'}

    def search(self,
               keyword: str,
               project_ids: List[str] = ["prd_jt", "x2_dev"],
               approved: bool = True):
        """
        Searches for annotation datasets based on the provided keyword and project IDs.

        Args:
            keyword (str): The keyword to search for in the dataset names.
            project_ids (List[str]): A list of project IDs to filter the search results. Default is ["prd_jt", "x2_dev"].
            approved (bool): Whether to filter by approved datasets. Default is True.

        Returns:
            List[str]: A list of dataset IDs that match the search criteria.
        """
        payload = {
            "approved": approved,
            "name_keyword": keyword,
            "project_ids": project_ids,
        }
        results = self._fetch_all_annotation_datasets(payload)
        return [data["id"] for data in results]

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
                url=self._base_url,
                headers=self._headers,
                params=params,
                data=json.dumps(payload))
            data = response.json()

            all_items.extend(data['items'])
            next_page_token = data.get('next_page_token')

            if not next_page_token:
                break

        return all_items


def update_database_config(config_file_path, keyword, project_ids):
    client = AnnotationDatasetSearchClient()
    found_datasets_ids = client.search(
        keyword=keyword, project_ids=project_ids)
    print(
        f"{len(found_datasets_ids)} found for keyword: {keyword} within projects: {project_ids}"
    )

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(config_file_path, 'r') as f:
        config_content = yaml.load(f)

    updated = False
    for dataset_id in found_datasets_ids:
        if dataset_id not in config_content['train'] + config_content[
                'val'] + config_content['test']:
            print(
                f"{dataset_id} seems to be a new ID, adding to the config file"
            )
            config_content['train'].append(dataset_id)
            updated = True

    if updated:
        # Ensure 'docs' field is treated as a literal block
        if 'docs' in config_content:
            config_content['docs'] = LiteralScalarString(config_content['docs'])

        with open(config_file_path, 'w') as f:
            yaml.dump(config_content, f)
        print('Config file updated with new datasets.')
    else:
        print('No new datasets found.')


def main():
    parser = argparse.ArgumentParser(
        description="Search for Annotation Datasets")
    parser.add_argument(
        "--config_path", type=str, required=True, help="Config path to update")
    parser.add_argument(
        "--keyword", type=str, required=True, help="The keyword to search for")
    parser.add_argument(
        "--project_ids",
        type=str,
        nargs="*",
        default=["prd_jt", "x2_dev"],
        help="List of project IDs to filter by (optional)")
    args = parser.parse_args()
    assert len(
        args.project_ids) > 0, "You MUST provide at least one project id."

    update_database_config(args.config_path, args.keyword, args.project_ids)


if __name__ == '__main__':
    main()
