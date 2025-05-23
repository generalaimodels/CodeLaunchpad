"""
Python Special Methods (Dunder Methods) Explained

This file covers descriptor methods and class creation methods in Python.
"""

###############################################################################
#                           DESCRIPTOR METHODS                                #
###############################################################################

"""
Descriptors are objects that implement at least one of the methods:
__get__, __set__, or __delete__. They provide a powerful way to customize
attribute access in Python classes.
"""

class Temperature:
    """
    A descriptor that converts between Celsius and Fahrenheit.
    Demonstrates the complete descriptor protocol.
    """
    def __init__(self, initial_value=0.0):
        self._value = float(initial_value)
    
    def __get__(self, instance, owner=None):
        """
        Gets called when accessing the attribute on an instance.
        
        Parameters:
            instance: The instance being accessed (or None for class access)
            owner: The class that owns the descriptor
            
        Returns:
            The attribute value when accessed from an instance
            The descriptor object itself when accessed from the class
        """
        if instance is None:  # Class access
            return self
        return self._value
    
    def __set__(self, instance, value):
        """
        Gets called when setting the attribute on an instance.
        
        Parameters:
            instance: The instance being modified
            value: The value being assigned
            
        This method allows validation, type conversion, or other
        custom behavior during attribute assignment.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Temperature must be a number")
        self._value = float(value)
    
    def __delete__(self, instance):
        """
        Gets called when deleting the attribute from an instance.
        
        Parameters:
            instance: The instance being modified
            
        This method can implement custom behavior when using:
        del instance.attribute
        """
        self._value = 0.0
    
    def __set_name__(self, owner, name):
        """
        Gets called when the descriptor is defined in a class.
        
        Parameters:
            owner: The class where the descriptor is defined
            name: The name of the attribute this descriptor is assigned to
            
        This method is called at class creation time, allowing the descriptor
        to know its attribute name, which is useful for storage or debugging.
        """
        self.name = name
        self.private_name = f"_{name}"


class WeatherStation:
    # Creating descriptor attributes
    current_temp = Temperature(20.0)
    yesterday_temp = Temperature(15.0)
    
    def __init__(self, location):
        self.location = location
    
    def __repr__(self):
        return f"WeatherStation(location='{self.location}')"


# Descriptor Usage Examples
def descriptor_examples():
    # Instantiate a weather station
    station = WeatherStation("New York")
    
    # __get__ gets called
    print(f"Current temperature: {station.current_temp}°C")
    
    # __set__ gets called
    station.current_temp = 25
    print(f"Updated temperature: {station.current_temp}°C")
    
    # __delete__ gets called
    del station.current_temp
    print(f"After deletion: {station.current_temp}°C")  # Will show 0.0
    
    # Class access calls __get__ with instance=None
    print(f"Descriptor object: {WeatherStation.current_temp}")


###############################################################################
#                           CLASS CREATION METHODS                            #
###############################################################################

"""
These methods are used in metaclasses to customize class creation and behavior.
"""

class OrderedMeta(type):
    """
    A metaclass that remembers the order of class attributes.
    Demonstrates __prepare__, __instancecheck__, and __subclasscheck__.
    """
    
    @classmethod
    def __prepare__(metacls, name, bases):
        """
        Prepares the namespace before the class body is executed.
        
        Parameters:
            metacls: The metaclass (OrderedMeta in this case)
            name: The name of the class being created
            bases: The base classes of the class being created
            
        Returns:
            A dictionary-like object to use as the namespace
            
        This method is called before class body execution to create the
        namespace where class attributes will be stored. By default, a 
        regular dict is used, but we can return custom mapping objects.
        """
        print(f"Preparing namespace for class {name}")
        # Return an ordered dictionary to preserve definition order
        return dict()
    
    def __new__(metacls, name, bases, namespace):
        """
        Creates and returns the new class object.
        
        This is included to show the complete metaclass workflow,
        though it's not one of the specific methods requested.
        """
        print(f"Creating class {name}")
        namespace['_attribute_order'] = list(namespace.keys())
        return super().__new__(metacls, name, bases, namespace)
    
    def __instancecheck__(cls, instance):
        """
        Controls the behavior of isinstance(instance, cls).
        
        Parameters:
            cls: The class being checked against
            instance: The object to check
            
        Returns:
            True if instance should be considered an instance of cls
            
        This method allows customizing what objects are considered
        instances of the class, beyond the normal inheritance rules.
        """
        # Default implementation, but we could add custom logic
        return super().__instancecheck__(instance)
    
    def __subclasscheck__(cls, subclass):
        """
        Controls the behavior of issubclass(subclass, cls).
        
        Parameters:
            cls: The class being checked against
            subclass: The class to check
            
        Returns:
            True if subclass should be considered a subclass of cls
            
        This method allows customizing what classes are considered
        subclasses, beyond the normal inheritance rules.
        """
        # Default implementation, but we could add custom logic
        return super().__subclasscheck__(subclass)


# Using the metaclass
class OrderedClass(metaclass=OrderedMeta):
    def method1(self):
        pass
    
    attribute1 = 1
    
    def method2(self):
        pass
    
    attribute2 = 2


# Virtual subclass example (demonstrates __instancecheck__ and __subclasscheck__)
class Shape:
    """Base class for shapes."""
    pass


class ShapeRegistry(type):
    """Metaclass to maintain a registry of shape types."""
    
    _registry = set()
    
    def __new__(metacls, name, bases, namespace):
        cls = super().__new__(metacls, name, bases, namespace)
        metacls._registry.add(cls)
        return cls
    
    def __instancecheck__(cls, instance):
        """
        Custom instanceof checks. Allows objects to be considered instances
        if they have a specific attribute structure, even without inheritance.
        """
        if super().__instancecheck__(instance):
            return True
        
        # Consider any object with area() and perimeter() methods a Shape
        has_area = hasattr(instance, 'area') and callable(instance.area)
        has_perimeter = hasattr(instance, 'perimeter') and callable(instance.perimeter)
        
        return has_area and has_perimeter
    
    def __subclasscheck__(cls, subclass):
        """
        Custom subclass checks. Allows classes to be considered subclasses
        even without direct inheritance.
        """
        if super().__subclasscheck__(subclass):
            return True
        
        if cls is Shape:
            # Check if subclass has appropriate methods
            has_area = hasattr(subclass, 'area') and callable(subclass.area)
            has_perimeter = hasattr(subclass, 'perimeter') and callable(subclass.perimeter)
            return has_area and has_perimeter
        
        return False


class Shape(metaclass=ShapeRegistry):
    """Base class for shapes with metaclass-based registry."""
    pass


class Rectangle:
    """Not explicitly inheriting from Shape"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)


def class_creation_examples():
    # Examining the attribute order stored by OrderedMeta
    print(f"OrderedClass attribute order: {OrderedClass._attribute_order}")
    
    # Testing __instancecheck__ and __subclasscheck__
    rect = Rectangle(5, 10)
    
    # This works even though Rectangle doesn't inherit from Shape
    print(f"Is rect a Shape instance? {isinstance(rect, Shape)}")
    print(f"Is Rectangle a Shape subclass? {issubclass(Rectangle, Shape)}")


# Main examples
if __name__ == "__main__":
    print("===== DESCRIPTOR EXAMPLES =====")
    descriptor_examples()
    
    print("\n===== CLASS CREATION EXAMPLES =====")
    class_creation_examples()