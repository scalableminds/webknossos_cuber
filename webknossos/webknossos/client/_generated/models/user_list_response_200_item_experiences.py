from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="UserListResponse200ItemExperiences")


@attr.s(auto_attribs=True)
class UserListResponse200ItemExperiences:
    """
    Attributes:
        abc (Union[Unset, int]):  Example: 5.
    """

    abc: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        abc = self.abc

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if abc is not UNSET:
            field_dict["abc"] = abc

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        abc = d.pop("abc", UNSET)

        user_list_response_200_item_experiences = cls(
            abc=abc,
        )

        user_list_response_200_item_experiences.additional_properties = d
        return user_list_response_200_item_experiences

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
