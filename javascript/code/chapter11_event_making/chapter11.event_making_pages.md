Okay boss, let's jump into Chapter 11: Events - Making Pages Interactive. This is where your web pages come alive, responding to user actions. We'll explore common browser events, how to listen for them, and how to handle them effectively. You'll become an event handling expert in no time!

**Chapter 11: Events - Making Pages Interactive**

**1. Common Browser Events**

Events are actions or occurrences that happen in the browser. Here are some of the most common ones:

| Event       | Description                                                                              | Trigger                                                                | Example                    |
| :---------- | :--------------------------------------------------------------------------------------- | :--------------------------------------------------------------------- | :------------------------- |
| `click`     | Occurs when an element is clicked                                                        | User clicks the element with the mouse                                 | Clicking a button         |
| `mouseover` | Occurs when the mouse cursor moves onto an element                                      | User moves the mouse over the element                                  | Hovering over a div       |
| `mouseout`  | Occurs when the mouse cursor moves out of an element                                    | User moves the mouse away from the element                             | Moving mouse out of a div |
| `keydown`   | Occurs when a key is pressed down (fires repeatedly if the key is held down)            | User presses any key on the keyboard                                   | Pressing an 'a' key        |
| `keyup`     | Occurs when a key is released                                                            | User releases the key on the keyboard                                   | Releasing 'a' key         |
| `focus`     | Occurs when an element gets focus                                                        | User clicks or uses tab key to select the element (input fields)     | Focusing an input field    |
| `blur`     | Occurs when an element looses focus                                                        | User clicks out side the element or focus changes to another element (input fields)     | Clicking outside the input field    |
| `submit`    | Occurs when a form is submitted                                                            | User clicks a submit button or presses Enter within a form            | Submitting a form         |
| `change`    | Occurs when value of a input field is changed                                                            | User edits the value of input field            | Changing input field     |
| `load` |Occurs when the page has finished loading            |After the web page has been fully loaded | Page load |

**2. Event Listeners**

To make your JavaScript code respond to events, you need to "listen" for those events. You do this by using the `addEventListener()` method on an HTML element.

```javascript
element.addEventListener(event, function, useCapture);
```

*   **`element`:** The HTML element where the event will occur.
*   **`event`:** A string representing the event type (e.g., `'click'`, `'mouseover'`).
*   **`function`:** A function to be executed when the event occurs (the event handler).
*   **`useCapture`:** An optional boolean value that specifies if the event should happen in the capturing phase or in the bubbling phase. (we can discuss in detail in advanced topic). The default is `false` (bubbling).

**Example:**

```html
<!DOCTYPE html>
<html>
<head>
    <title>Event Handling</title>
</head>
<body>
    <button id="myButton">Click Me</button>
    <div id="myDiv">Hover Over Me</div>
    <input type="text" id="myInput">
    <form id = "myForm">
        <input type="submit" value="Submit" />
    </form>
    <script>
        let button = document.getElementById('myButton');
        let div = document.getElementById('myDiv');
        let input = document.getElementById('myInput');
        let form = document.getElementById('myForm')

        button.addEventListener('click', function() {
            console.log("Button Clicked!");
        });

        div.addEventListener('mouseover', function() {
            console.log("Mouse Over!");
        });

        div.addEventListener('mouseout', function() {
          console.log("Mouse out!");
        });

        input.addEventListener('focus', function() {
            console.log("Input field is focused")
        })

        input.addEventListener('blur', function() {
            console.log("Input field lost focus")
        })

        input.addEventListener('keydown', function(event) {
            console.log("Key Down: " + event.key); // event.key gives which key is pressed
        });

        input.addEventListener('keyup', function(event) {
            console.log("Key Up: " + event.key);
        });

        form.addEventListener('submit', function(event) {
            event.preventDefault() // to prevent form submission
            console.log("Form is Submitted");
        })

    </script>
</body>
</html>
```

**Important Notes:**

*   The function you pass to the `addEventListener()` will be executed when the specific event occurs on the given element.
*   You can add multiple event listeners to the same element for the same or different events.
*   The function used in event listener will be called *event handler*
*   We are using anonymous function as an event handler here. It is possible to define a function separately, and pass it as an event handler to the event listener.

**3. Event Handling**

*   When an event occurs, the browser creates an *event object* and passes it as an argument to your event handler function. This object contains information about the event (e.g., which element triggered the event, which key was pressed, mouse position, etc.).

*   In the above example, we can access the event object as an argument of the callback function like `function(event) {}`

*   The `event` object has properties like:
    *  `event.target`: reference to element that triggered the event
    *  `event.key`: which key was pressed.
    *  `event.preventDefault()`: used to prevent the default action of event.
      *   For example, the default action of form submit is to refresh the page. Using `event.preventDefault()` we can stop that behaviour.
      *   `event.stopPropagation()`: used to stop event propagation (will discuss more in advanced topic).

**Example (from your instructions):**

```html
<!DOCTYPE html>
<html>
<head>
    <title>Event Example</title>
</head>
<body>
    <button id="myButton">Click Me</button>
    <script>
        let button = document.getElementById('myButton');
        button.addEventListener('click', function() {
            console.log("Button Clicked!");
        });
    </script>
</body>
</html>
```

*   This example shows how to use `addEventListener` to listen for a `click` event on a button.

**Expected Outcome:**

You should now be able to:

*   Understand common browser events and when they occur.
*   Use `addEventListener` to listen for events on HTML elements.
*   Create event handlers that respond to user interactions.
*   Access and use the event object to get information about the event.
*  Make interactive web pages.

That wraps up Chapter 11, boss! You are now an event handling master, able to make your web pages truly interactive. Practice with different events and event handlers. Any questions, please feel free to ask. We are now ready for asynchronous javascript! Let's go! 🚀
