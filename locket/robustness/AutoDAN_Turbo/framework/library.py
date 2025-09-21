# ruff: noqa: E501

import json


class Library:
    def __init__(self, library, logger):
        self.library = library
        self.logger = logger

    def merge(self, dict1, dict2):
        if not dict1:
            return dict2
        merged = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())
        for key in all_keys:
            if key in dict1 and key in dict2:
                merged[key] = dict1[key]
                if (
                    "Example" in dict1[key]
                    and "Example" in dict2[key]
                    and dict2[key]["Example"]
                ):
                    dict1[key]["Example"].append(dict2[key]["Example"][0])
                    dict1[key]["Score"].append(dict2[key]["Score"][0])
                    dict1[key]["Embeddings"].append(dict2[key]["Embeddings"][0])
            elif key in dict1:
                merged[key] = dict1[key]
            else:
                merged[key] = dict2[key]
        return merged

    def add(self, new_strategy, if_notify=False):
        """
        :param new_strategy: a dictionary containing the new strategy to be added to the library
        """
        try:
            # Normalize and ensure required keys
            name = new_strategy.get("Strategy") or "Unnamed Strategy"
            if "Definition" not in new_strategy or not isinstance(new_strategy["Definition"], str):
                new_strategy["Definition"] = ""
            new_strategy = {name: {**new_strategy, "Strategy": name}}
            self.library = self.merge(self.library, new_strategy)
            if if_notify:
                for key, value_dict in new_strategy.items():
                    new_dict = {
                        "Strategy": value_dict.get("Strategy", ""),
                        "Definition": value_dict.get("Definition", ""),
                    }
                    self.logger.info(
                        f"New strategy added: {json.dumps(new_dict, indent=4, ensure_ascii=False)}"
                    )
        except Exception as e:
            self.logger.error(f"Failed to add strategy to library: {e}")

    def all(self):
        return self.library
