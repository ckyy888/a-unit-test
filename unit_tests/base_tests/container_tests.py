"""
Container Test Mixin
Purpose: Verify container membership and relationships
"""
from typing import Any, Callable, Dict, List, Optional, Union, Set


class ContainerTestMixin:
    """Mixin for testing container membership and relationships."""
    
    def assertInContainer(self, item: Any, container: Union[List[Any], Set[Any], Dict[Any, Any]], 
                         msg: Optional[str] = None) -> None:
        """
        Verify that an item is in a container.
        
        Args:
            item: Item to check for membership
            container: Container to check (list, set, or dict)
            msg: Optional custom message
        """
        if isinstance(container, dict):
            # For dictionaries, check if item is a key
            self.assertIn(item, container.keys(), msg)
        else:
            self.assertIn(item, container, msg)
    
    def assertNotInContainer(self, item: Any, container: Union[List[Any], Set[Any], Dict[Any, Any]], 
                           msg: Optional[str] = None) -> None:
        """
        Verify that an item is NOT in a container.
        
        Args:
            item: Item to check for membership
            container: Container to check (list, set, or dict)
            msg: Optional custom message
        """
        if isinstance(container, dict):
            # For dictionaries, check if item is a key
            self.assertNotIn(item, container.keys(), msg)
        else:
            self.assertNotIn(item, container, msg)
    
    def assertContainerContains(self, expected_items: List[Any], 
                              container: Union[List[Any], Set[Any], Dict[Any, Any]]) -> None:
        """
        Verify that a container contains all expected items.
        
        Args:
            expected_items: List of items that should be in the container
            container: Container to check
        """
        if isinstance(container, dict):
            container_items = list(container.keys())
        else:
            container_items = list(container)
        
        for item in expected_items:
            self.assertIn(item, container_items,
                         f"Expected item '{item}' not found in container")
    
    def assertContainerNotContains(self, unexpected_items: List[Any], 
                                 container: Union[List[Any], Set[Any], Dict[Any, Any]]) -> None:
        """
        Verify that a container does NOT contain any of the unexpected items.
        
        Args:
            unexpected_items: List of items that should NOT be in the container
            container: Container to check
        """
        if isinstance(container, dict):
            container_items = list(container.keys())
        else:
            container_items = list(container)
        
        for item in unexpected_items:
            self.assertNotIn(item, container_items,
                           f"Unexpected item '{item}' found in container")
    
    def assertContainerSize(self, expected_size: int, 
                          container: Union[List[Any], Set[Any], Dict[Any, Any]]) -> None:
        """
        Verify that a container has the expected size.
        
        Args:
            expected_size: Expected number of items in the container
            container: Container to check
        """
        actual_size = len(container)
        self.assertEqual(expected_size, actual_size,
                        f"Container size mismatch: expected {expected_size}, got {actual_size}")
    
    def assertContainerEmpty(self, container: Union[List[Any], Set[Any], Dict[Any, Any]]) -> None:
        """
        Verify that a container is empty.
        
        Args:
            container: Container to check
        """
        self.assertEqual(len(container), 0,
                        f"Container is not empty: contains {len(container)} items")
    
    def assertContainerNotEmpty(self, container: Union[List[Any], Set[Any], Dict[Any, Any]]) -> None:
        """
        Verify that a container is not empty.
        
        Args:
            container: Container to check
        """
        self.assertGreater(len(container), 0,
                          "Container is empty but should not be")
    
    def assertContainerSubset(self, expected_subset: Union[List[Any], Set[Any]], 
                            container: Union[List[Any], Set[Any], Dict[Any, Any]]) -> None:
        """
        Verify that a container contains all items from the expected subset.
        
        Args:
            expected_subset: Expected subset of items
            container: Container to check
        """
        if isinstance(container, dict):
            container_items = set(container.keys())
        else:
            container_items = set(container)
        
        expected_items = set(expected_subset)
        self.assertTrue(expected_items.issubset(container_items),
                       f"Container does not contain expected subset: {expected_items}")
    
    def assertContainerSuperset(self, expected_superset: Union[List[Any], Set[Any]], 
                              container: Union[List[Any], Set[Any], Dict[Any, Any]]) -> None:
        """
        Verify that a container is a subset of the expected superset.
        
        Args:
            expected_superset: Expected superset of items
            container: Container to check
        """
        if isinstance(container, dict):
            container_items = set(container.keys())
        else:
            container_items = set(container)
        
        expected_items = set(expected_superset)
        self.assertTrue(container_items.issubset(expected_items),
                       f"Container is not a subset of expected superset: {container_items}")
    
    def assertContainerEquals(self, expected_container: Union[List[Any], Set[Any], Dict[Any, Any]], 
                            actual_container: Union[List[Any], Set[Any], Dict[Any, Any]]) -> None:
        """
        Verify that two containers are equal.
        
        Args:
            expected_container: Expected container
            actual_container: Actual container
        """
        if isinstance(expected_container, dict) and isinstance(actual_container, dict):
            self.assertEqual(set(expected_container.keys()), set(actual_container.keys()),
                           "Container keys don't match")
            for key in expected_container:
                self.assertEqual(expected_container[key], actual_container[key],
                               f"Container value mismatch for key '{key}'")
        else:
            # Convert to sets for comparison (ignores order)
            expected_set = set(expected_container)
            actual_set = set(actual_container)
            self.assertEqual(expected_set, actual_set,
                           f"Container contents don't match: expected {expected_set}, got {actual_set}")
    
    def assertContainerOrder(self, expected_order: List[Any], 
                           container: List[Any]) -> None:
        """
        Verify that items in a container appear in the expected order.
        
        Args:
            expected_order: Expected order of items
            container: Container to check (must be a list)
        """
        if not isinstance(container, list):
            self.fail("Container must be a list to check order")
        
        self.assertEqual(len(expected_order), len(container),
                        f"Container length mismatch: expected {len(expected_order)}, got {len(container)}")
        
        for i, (expected, actual) in enumerate(zip(expected_order, container)):
            self.assertEqual(expected, actual,
                           f"Container order mismatch at position {i}: expected {expected}, got {actual}")
    
    def assertContainerUnique(self, container: Union[List[Any], Set[Any], Dict[Any, Any]]) -> None:
        """
        Verify that all items in a container are unique.
        
        Args:
            container: Container to check
        """
        if isinstance(container, dict):
            items = list(container.keys())
        else:
            items = list(container)
        
        unique_items = set(items)
        self.assertEqual(len(items), len(unique_items),
                        f"Container contains duplicate items: {items}")
    
    def assertContainerSorted(self, container: List[Any], 
                            reverse: bool = False) -> None:
        """
        Verify that items in a container are sorted.
        
        Args:
            container: Container to check (must be a list)
            reverse: Whether items should be sorted in reverse order
        """
        if not isinstance(container, list):
            self.fail("Container must be a list to check sorting")
        
        sorted_items = sorted(container, reverse=reverse)
        self.assertEqual(container, sorted_items,
                        f"Container is not sorted: {container}")
    
    def assertContainerFiltered(self, container: Union[List[Any], Set[Any]], 
                              filter_condition: Callable[[Any], bool]) -> None:
        """
        Verify that all items in a container satisfy a filter condition.
        
        Args:
            container: Container to check
            filter_condition: Function that returns True for items that should be in the container
        """
        for item in container:
            self.assertTrue(filter_condition(item),
                          f"Item '{item}' does not satisfy filter condition")
    
    def assertContainerMapped(self, container: Union[List[Any], Set[Any]], 
                            map_function: Callable[[Any], Any], 
                            expected_results: Union[List[Any], Set[Any]]) -> None:
        """
        Verify that mapping a function over a container produces expected results.
        
        Args:
            container: Container to map over
            map_function: Function to apply to each item
            expected_results: Expected results after mapping
        """
        actual_results = list(map(map_function, container))
        expected_list = list(expected_results)
        
        self.assertEqual(len(actual_results), len(expected_list),
                        f"Mapping result length mismatch: expected {len(expected_list)}, got {len(actual_results)}")
        
        for i, (expected, actual) in enumerate(zip(expected_list, actual_results)):
            self.assertEqual(expected, actual,
                           f"Mapping result mismatch at position {i}: expected {expected}, got {actual}")
    
    def assertContainerReduced(self, container: Union[List[Any], Set[Any]], 
                             reduce_function: Callable[[Any, Any], Any], 
                             expected_result: Any, 
                             initial_value: Optional[Any] = None) -> None:
        """
        Verify that reducing a container produces the expected result.
        
        Args:
            container: Container to reduce
            reduce_function: Function to apply for reduction
            expected_result: Expected result after reduction
            initial_value: Optional initial value for reduction
        """
        if initial_value is not None:
            actual_result = reduce(reduce_function, container, initial_value)
        else:
            actual_result = reduce(reduce_function, container)
        
        self.assertEqual(expected_result, actual_result,
                        f"Reduction result mismatch: expected {expected_result}, got {actual_result}")
    
    def assertContainerAll(self, container: Union[List[Any], Set[Any]], 
                          predicate: Callable[[Any], bool]) -> None:
        """
        Verify that all items in a container satisfy a predicate.
        
        Args:
            container: Container to check
            predicate: Function that returns True for items that should satisfy the condition
        """
        for item in container:
            self.assertTrue(predicate(item),
                          f"Item '{item}' does not satisfy predicate")
    
    def assertContainerAny(self, container: Union[List[Any], Set[Any]], 
                          predicate: Callable[[Any], bool]) -> None:
        """
        Verify that at least one item in a container satisfies a predicate.
        
        Args:
            container: Container to check
            predicate: Function that returns True for items that should satisfy the condition
        """
        satisfied = any(predicate(item) for item in container)
        self.assertTrue(satisfied,
                       f"No item in container satisfies predicate: {container}")
    
    def assertContainerNone(self, container: Union[List[Any], Set[Any]], 
                           predicate: Callable[[Any], bool]) -> None:
        """
        Verify that no items in a container satisfy a predicate.
        
        Args:
            container: Container to check
            predicate: Function that returns True for items that should NOT satisfy the condition
        """
        for item in container:
            self.assertFalse(predicate(item),
                           f"Item '{item}' satisfies predicate but should not")
    
    def assertContainerCount(self, item: Any, expected_count: int, 
                           container: Union[List[Any], Set[Any], Dict[Any, Any]]) -> None:
        """
        Verify that an item appears the expected number of times in a container.
        
        Args:
            item: Item to count
            expected_count: Expected number of occurrences
            container: Container to check
        """
        if isinstance(container, dict):
            actual_count = container.get(item, 0)
        else:
            actual_count = container.count(item)
        
        self.assertEqual(expected_count, actual_count,
                        f"Item '{item}' count mismatch: expected {expected_count}, got {actual_count}")
    
    def assertContainerMin(self, expected_min: Any, 
                          container: Union[List[Any], Set[Any]]) -> None:
        """
        Verify that the minimum value in a container is as expected.
        
        Args:
            expected_min: Expected minimum value
            container: Container to check
        """
        if not container:
            self.fail("Cannot find minimum of empty container")
        
        actual_min = min(container)
        self.assertEqual(expected_min, actual_min,
                        f"Container minimum mismatch: expected {expected_min}, got {actual_min}")
    
    def assertContainerMax(self, expected_max: Any, 
                          container: Union[List[Any], Set[Any]]) -> None:
        """
        Verify that the maximum value in a container is as expected.
        
        Args:
            expected_max: Expected maximum value
            container: Container to check
        """
        if not container:
            self.fail("Cannot find maximum of empty container")
        
        actual_max = max(container)
        self.assertEqual(expected_max, actual_max,
                        f"Container maximum mismatch: expected {expected_max}, got {actual_max}")
    
    def assertContainerSum(self, expected_sum: Any, 
                          container: Union[List[Any], Set[Any]]) -> None:
        """
        Verify that the sum of all items in a container is as expected.
        
        Args:
            expected_sum: Expected sum
            container: Container to check
        """
        actual_sum = sum(container)
        self.assertEqual(expected_sum, actual_sum,
                        f"Container sum mismatch: expected {expected_sum}, got {actual_sum}")
    
    def assertContainerAverage(self, expected_average: float, 
                             container: Union[List[Any], Set[Any]]) -> None:
        """
        Verify that the average of all items in a container is as expected.
        
        Args:
            expected_average: Expected average
            container: Container to check
        """
        if not container:
            self.fail("Cannot find average of empty container")
        
        actual_average = sum(container) / len(container)
        self.assertAlmostEqual(expected_average, actual_average, places=6,
                              msg=f"Container average mismatch: expected {expected_average}, got {actual_average}")
    
    def assertContainerIntersection(self, container1: Union[List[Any], Set[Any]], 
                                  container2: Union[List[Any], Set[Any]], 
                                  expected_intersection: Union[List[Any], Set[Any]]) -> None:
        """
        Verify that the intersection of two containers is as expected.
        
        Args:
            container1: First container
            container2: Second container
            expected_intersection: Expected intersection
        """
        set1 = set(container1)
        set2 = set(container2)
        actual_intersection = set1.intersection(set2)
        expected_set = set(expected_intersection)
        
        self.assertEqual(expected_set, actual_intersection,
                        f"Container intersection mismatch: expected {expected_set}, got {actual_intersection}")
    
    def assertContainerUnion(self, container1: Union[List[Any], Set[Any]], 
                           container2: Union[List[Any], Set[Any]], 
                           expected_union: Union[List[Any], Set[Any]]) -> None:
        """
        Verify that the union of two containers is as expected.
        
        Args:
            container1: First container
            container2: Second container
            expected_union: Expected union
        """
        set1 = set(container1)
        set2 = set(container2)
        actual_union = set1.union(set2)
        expected_set = set(expected_union)
        
        self.assertEqual(expected_set, actual_union,
                        f"Container union mismatch: expected {expected_set}, got {actual_union}")
    
    def assertContainerDifference(self, container1: Union[List[Any], Set[Any]], 
                                container2: Union[List[Any], Set[Any]], 
                                expected_difference: Union[List[Any], Set[Any]]) -> None:
        """
        Verify that the difference of two containers is as expected.
        
        Args:
            container1: First container
            container2: Second container
            expected_difference: Expected difference (items in container1 but not in container2)
        """
        set1 = set(container1)
        set2 = set(container2)
        actual_difference = set1.difference(set2)
        expected_set = set(expected_difference)
        
        self.assertEqual(expected_set, actual_difference,
                        f"Container difference mismatch: expected {expected_set}, got {actual_difference}")
    
    def assertContainerSymmetricDifference(self, container1: Union[List[Any], Set[Any]], 
                                         container2: Union[List[Any], Set[Any]], 
                                         expected_symmetric_difference: Union[List[Any], Set[Any]]) -> None:
        """
        Verify that the symmetric difference of two containers is as expected.
        
        Args:
            container1: First container
            container2: Second container
            expected_symmetric_difference: Expected symmetric difference
        """
        set1 = set(container1)
        set2 = set(container2)
        actual_symmetric_difference = set1.symmetric_difference(set2)
        expected_set = set(expected_symmetric_difference)
        
        self.assertEqual(expected_set, actual_symmetric_difference,
                        f"Container symmetric difference mismatch: expected {expected_set}, got {actual_symmetric_difference}")
    
    def assertContainerDisjoint(self, container1: Union[List[Any], Set[Any]], 
                              container2: Union[List[Any], Set[Any]]) -> None:
        """
        Verify that two containers are disjoint (have no common elements).
        
        Args:
            container1: First container
            container2: Second container
        """
        set1 = set(container1)
        set2 = set(container2)
        
        self.assertTrue(set1.isdisjoint(set2),
                       f"Containers are not disjoint: {set1.intersection(set2)}")
    
    def assertContainerOverlap(self, container1: Union[List[Any], Set[Any]], 
                             container2: Union[List[Any], Set[Any]], 
                             min_overlap: int = 1) -> None:
        """
        Verify that two containers have at least a minimum number of overlapping elements.
        
        Args:
            container1: First container
            container2: Second container
            min_overlap: Minimum number of overlapping elements required
        """
        set1 = set(container1)
        set2 = set(container2)
        actual_overlap = len(set1.intersection(set2))
        
        self.assertGreaterEqual(actual_overlap, min_overlap,
                               f"Container overlap too small: {actual_overlap} < {min_overlap}")
    
    def assertContainerInvariant(self, invariant_condition: Callable[[Union[List[Any], Set[Any], Dict[Any, Any]]], bool], 
                               container: Union[List[Any], Set[Any], Dict[Any, Any]]) -> None:
        """
        Verify that a container satisfies a custom invariant condition.
        
        Args:
            invariant_condition: Function that takes a container and returns True if invariant holds
            container: Container to check
        """
        self.assertTrue(invariant_condition(container),
                       f"Container invariant violated: {container}") 