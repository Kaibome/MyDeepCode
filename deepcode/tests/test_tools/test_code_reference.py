import json
import os
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from deepcode.tools.code_reference_indexer import (  # noqa: E402
    CodeReference,
    RelationshipInfo,
    calculate_relevance_score,
    extract_code_references,
    get_indexes_overview,
    load_index_files_from_directory,
    search_code_references,
)


class TestCodeReferenceIndexer(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.mock_indexes_dir = os.path.join(self.test_dir, "mock_indexes")
        self.mock_index_file_path = os.path.join(
            self.mock_indexes_dir, "mock_repo_index.json"
        )
        os.makedirs(self.mock_indexes_dir, exist_ok=True)

        mock_index_data = {
            "repo_name": "mock_repo",
            "total_files": 2,
            "file_summaries": [
                {
                    "file_path": "src/implementation.py",
                    "file_type": "Python module",
                    "main_functions": ["process_data", "analyze_results"],
                    "key_concepts": ["data processing", "analysis"],
                    "dependencies": ["numpy", "pandas"],
                    "summary": "Implementation of data processing functionality",
                    "lines_of_code": 100,
                }
            ],
            "relationships": [
                {
                    "repo_file_path": "src/implementation.py",
                    "target_file_path": "test_implementation.py",
                    "relationship_type": "direct_match",
                    "confidence_score": 0.9,
                    "helpful_aspects": ["implementation pattern"],
                    "potential_contributions": ["core functionality"],
                    "usage_suggestions": "Use this as a reference",
                }
            ],
        }
        with open(self.mock_index_file_path, "w", encoding="utf-8") as f:
            json.dump(mock_index_data, f)

    def tearDown(self):
        if os.path.exists(self.mock_index_file_path):
            os.remove(self.mock_index_file_path)
        if os.path.exists(self.mock_indexes_dir):
            os.rmdir(self.mock_indexes_dir)

    def test_code_reference_dataclass(self):
        ref = CodeReference(
            file_path="test.py",
            file_type="Python module",
            main_functions=["main", "helper"],
            key_concepts=["algorithm", "data processing"],
            dependencies=["numpy", "pandas"],
            summary="Test file",
            lines_of_code=42,
            repo_name="test-repo",
            confidence_score=0.8,
        )
        self.assertEqual(ref.file_path, "test.py")
        self.assertEqual(ref.repo_name, "test-repo")

    def test_relationship_info_dataclass(self):
        rel = RelationshipInfo(
            repo_file_path="file1.py",
            target_file_path="file2.py",
            relationship_type="direct_match",
            confidence_score=0.9,
            helpful_aspects=["implementation"],
            potential_contributions=["core functionality"],
            usage_suggestions="Use this file for core implementation",
        )
        self.assertEqual(rel.target_file_path, "file2.py")
        self.assertEqual(rel.relationship_type, "direct_match")

    def test_load_index_files_from_directory(self):
        index_cache = load_index_files_from_directory(self.mock_indexes_dir)
        self.assertEqual(len(index_cache), 1)

        non_existent_dir = os.path.join(self.test_dir, "non_existent_dir")
        index_cache = load_index_files_from_directory(non_existent_dir)
        self.assertEqual(len(index_cache), 0)

    def test_extract_code_references(self):
        index_data = {
            "repo_name": "test-repo",
            "file_summaries": [
                {
                    "file_path": "test.py",
                    "file_type": "Python module",
                    "main_functions": ["main"],
                    "key_concepts": ["test"],
                    "dependencies": [],
                    "summary": "Test file",
                    "lines_of_code": 1,
                }
            ],
        }
        references = extract_code_references(index_data)
        self.assertEqual(len(references), 1)
        self.assertEqual(references[0].repo_name, "test-repo")

    def test_calculate_relevance_score(self):
        ref = CodeReference(
            file_path="implementation.py",
            file_type="Python module",
            main_functions=["process_data"],
            key_concepts=["data processing"],
            dependencies=[],
            summary="Implementation file",
            lines_of_code=100,
            repo_name="test-repo",
        )
        score = calculate_relevance_score("test_implementation.py", ref)
        self.assertGreater(score, 0.0)

    @patch("deepcode.tools.code_reference_indexer.load_index_files_from_directory")
    async def test_search_code_references(self, mock_load_indexes):
        mock_load_indexes.return_value = {
            "mock_repo": {
                "repo_name": "mock_repo",
                "file_summaries": [
                    {
                        "file_path": "implementation.py",
                        "file_type": "Python module",
                        "main_functions": ["process_data"],
                        "key_concepts": ["data processing"],
                        "dependencies": [],
                        "summary": "Implementation file",
                        "lines_of_code": 100,
                    }
                ],
                "relationships": [],
            }
        }

        result_json = await search_code_references(
            indexes_path="dummy_path",
            target_file="test_implementation.py",
            keywords="data,process",
        )
        result = json.loads(result_json)
        self.assertEqual(result["status"], "success")

    @patch("deepcode.tools.code_reference_indexer.load_index_files_from_directory")
    async def test_get_indexes_overview(self, mock_load_indexes):
        mock_load_indexes.return_value = {
            "mock_repo": {
                "repo_name": "mock_repo",
                "total_files": 2,
                "file_summaries": [
                    {"file_path": "file1.py", "file_type": "Python module", "key_concepts": ["a"]},
                    {"file_path": "file2.py", "file_type": "Python script", "key_concepts": ["b"]},
                ],
                "relationships": [{}, {}],
            }
        }
        result_json = await get_indexes_overview(indexes_path="dummy_path")
        result = json.loads(result_json)
        self.assertEqual(result["status"], "success")


if __name__ == "__main__":
    unittest.main()
