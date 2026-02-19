# Copyright (c) 2025 Regents of University of Michigan
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
WSTL Base Module

Shared mixin classes used by both the wstlpy and wstltorch
implementations.

Classes:
    WSTLKeyMixin: key-hash bookkeeping shared by WSTLFormula and WSTLFormulaTorch
    UnaryMixin: key setting and string representation for unary operators
    BinaryMixin: key setting and string representation for binary operators
    PredicateMixin: key setting and string representation for predicates
    LessThanMixin: robustness computation and operator name for LessThan predicate
    GreaterThanMixin: robustness computation and operator name for GreaterThan predicate
    EqualMixin: operator name for Equal predicate
"""


class WSTLKeyMixin:
    """
    Mixin providing key and hash management for WSTL formula objects.

    Intended to be used as a base class (via multiple inheritance) for both
    WSTLFormula and WSTLFormulaTorch. It provides no __init__;
    subclasses are responsible for initialising self.keys, self.key_hashes,
    and self.key.
    """

    def set_keys(self) -> None:
        """Set the list of weight-dictionary keys for this formula."""
        self.keys = []
        self.keys = self._set_key(self.keys)

    def set_key_hash(self) -> None:
        """Populate ``self.key_hashes`` with the hash of each key."""
        self.key_hashes = []
        for key in self.keys:
            self.key_hashes.append(str(hash(key)))

    @property
    def key_hash(self) -> str:
        """Hash of this formula's own key."""
        return str(hash(self.key))

    def __repr__(self) -> str:
        return self.__str__()


class UnaryMixin:
    """
    Mixin for unary temporal operators (Always, Eventually)

    Requires that the host class provides:
    * self.subformula subformula with its own _set_key method
    * self.interval: Interval instance with a __str__ method
    * self.operation_name: string operator symbol (e.g. "G" or "F")
    """

    def _set_key(self, key_list: list) -> list:
        """Set dictionary names for the operator and its subformula."""
        if not isinstance(key_list, list):
            raise TypeError(f"key_list should be a list, got {type(key_list)}.")
        key_list = self.subformula._set_key(key_list)
        key_list = self._set_operator_key(key_list)
        return key_list

    def _set_operator_key(self, key_list: list) -> list:
        """Compute and register the unique key for this operator."""
        interval_str = str(self.interval).replace(".", "_")
        key = f"{self.operation_name}{interval_str}({self.subformula.key})"

        # if there's a duplicate key already in the list,
        # e.g. always phi and always phi, append underscores until it's unique
        while key in key_list:
            key += "_"

        self.key = key
        key_list.append(self.key)
        return key_list

    def __str__(self) -> str:
        return f"{self.operation_name}{str(self.interval)}({self.subformula})"


class BinaryMixin:
    """s
    Mixin for binary logical operators (And, Or).

    Requires that the host class provides:
    * self.subformula1, self.subformula2: subformulas with _set_key
    * self.operation_name: string operator symbol (e.g. "and" or "or")
    """

    def _set_key(self, key_list: list) -> list:
        """Set dictionary names for the operator and both subformulas."""
        if not isinstance(key_list, list):
            raise TypeError(f"key_list should be a list, got {type(key_list)}.")
        key_list = self.subformula1._set_key(key_list)
        key_list = self.subformula2._set_key(key_list)
        key_list = self._set_operator_key(key_list)
        return key_list

    def _set_operator_key(self, key_list: list) -> list:
        """Compute and register the unique key for this operator."""
        key = f"({self.subformula1.key}){self.operation_name}({self.subformula2.key})"

        while key in key_list:
            key += "_"

        self.key = key
        key_list.append(self.key)
        return key_list

    def __str__(self) -> str:
        return f"({self.subformula1}){self.operation_name}({self.subformula2})"


class PredicateMixin:
    """
    Mixin for atomic predicates (LessThan, GreaterThan, Equal).

    Requires that the host class provides:
    * self.signal: Signal instance with a name attribute
    * self.threshold: numeric threshold value
    * self.operation_name: string operator symbol (e.g. "<" or ">")
    """

    def _set_key(self, key_list: list) -> list:
        """Set dictionary name for this predicate."""
        key = str(self).replace(".", "_")

        while key in key_list:
            key += "_"

        self.key = key
        key_list.append(self.key)
        return key_list

    def __str__(self):
        """Returns a string representation of the LessThan instance."""
        return f"{self.signals}{self.operation_name}{self.value}"

    def __repr__(self):
        return self.__str__()


class LessThanMixin(PredicateMixin):
    """Mixin for LessThan predicate."""

    @property
    def operation_name(self):
        return "<="

    @property
    def comparison_op(self):
        return "<="

    def _operator_robustness(self, trace):
        """Robustness: value - trace  (works for numpy arrays and torch tensors)."""
        return self.value - trace


class GreaterThanMixin(PredicateMixin):
    """Mixin for GreaterThan predicate."""

    @property
    def operation_name(self):
        return ">="

    @property
    def comparison_op(self):
        return ">="

    def _operator_robustness(self, trace):
        """Robustness: trace - value  (works for numpy arrays and torch tensors)."""
        return trace - self.value


class EqualMixin(PredicateMixin):
    """Mixin for Equal predicate."""

    @property
    def operation_name(self):
        return "=="

    @property
    def comparison_op(self):
        return "=="

    def _operator_robustness(self, trace):
        """Robustness: -|trace - value|.

        Uses the built-in abs() which dispatches to __abs__ on both
        numpy arrays and torch tensors, keeping this backend-agnostic.
        """
        return -abs(trace - self.value)
