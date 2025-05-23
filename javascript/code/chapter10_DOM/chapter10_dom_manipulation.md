Okay boss, let's get into Chapter 10: DOM Manipulation (Document Object Model). This is where JavaScript starts interacting with your HTML pages, making them dynamic and interactive. We'll cover everything from understanding the DOM to manipulating elements, so you'll become a DOM master!

**Chapter 10: DOM Manipulation (Document Object Model)**

**1. Introduction to the DOM Tree**

*   **DOM (Document Object Model):** The DOM is a programming interface for HTML documents. When a browser loads an HTML page, it creates a model of the page as a tree-like structure (DOM tree) in memory.
*   **DOM Tree:** It's a hierarchical representation of the HTML elements in a webpage. The top-level element is the `document` object, which represents the entire HTML document.
    *   The `document` has child elements like `html`, `head`, `body`, and so on.
    *   These child elements can have further child elements of their own, forming a hierarchical tree-like structure.
*   JavaScript can access this DOM tree and change elements, styles and more, thus, making the web page dynamic.

Here's a simplified visualization:

```
       document
           |
         <html>
           |   |
        <head>  <body>
          |       |     |
       <title>   <h1>  <p>
                    |
                  "Title"
```

**2. Selecting HTML Elements**

To manipulate HTML elements, you first need to select them. JavaScript provides several methods for doing so using the `document` object:

| Method                     | Description                                                                | Example                                         |
| :------------------------- | :------------------------------------------------------------------------- | :---------------------------------------------- |
| `getElementById(id)`      | Selects an element with the given `id` attribute. Should be unique for every element. | `document.getElementById('myHeading')`          |
| `getElementsByClassName(className)` | Selects all elements with the given `class` name. Returns an HTMLCollection (array like object). | `document.getElementsByClassName('myParagraph')` |
| `getElementsByTagName(tagName)`     |Selects all element with given tag name. Returns HTMLCollection     |`document.getElementsByTagName('p')`             |
| `querySelector(selector)`   | Selects the first element that matches the given CSS selector (like in CSS). | `document.querySelector('#myList li:first-child')` |
| `querySelectorAll(selector)`| Selects all elements that match the given CSS selector. Returns a NodeList (array like object). | `document.querySelectorAll('.myListItem')`        |

*   `getElementById()` selects element by `id` attribute
    * Id's are unique, so only single element will be selected.
*   `getElementsByClassName()` selects element by `class` attribute.
    *  Class can be assigned to multiple elements, hence multiple elements can be selected. Returns HTML Collection.
*  `getElementsByTagName()` selects elements by `tag` attribute.
     * There can be multiple elements with same tag. Returns HTML Collection
*  `querySelector()` selects element based on css selector query
    *  Return only the first element that match the query.
*  `querySelectorAll()` selects element based on css selector query
    *   Return all elements that match the query.

**Example:**

```html
<!DOCTYPE html>
<html>
<head>
    <title>DOM Manipulation</title>
</head>
<body>
    <h1 id="main-heading">Hello, DOM!</h1>
    <p class="my-paragraph">This is a paragraph.</p>
    <p class="my-paragraph">Another paragraph.</p>
    <ul id="my-list">
      <li class = "my-list-item">Item 1</li>
      <li class = "my-list-item">Item 2</li>
      <li class = "my-list-item">Item 3</li>
    </ul>
    <script>
      let heading = document.getElementById('main-heading');
      let paragraphs = document.getElementsByClassName('my-paragraph');
      let list = document.getElementById('my-list')
      let listItems = document.querySelectorAll('.my-list-item')

      console.log(heading)
      console.log(paragraphs) // Output: HTMLCollection (array like object)
      console.log(list)
      console.log(listItems) // Output: NodeList (array like object)
    </script>
</body>
</html>
```
**3. Modifying HTML Elements**

Once you have selected an element, you can modify it using various properties:

| Property/Method   | Description                                                   | Example                                  |
| :---------------- | :------------------------------------------------------------ | :--------------------------------------- |
| `innerHTML`       | Gets or sets the HTML content of an element (can be dangerous to use user input) | `element.innerHTML = '<b>New text</b>'`    |
| `textContent`     | Gets or sets the text content of an element (safer than `innerHTML`)  | `element.textContent = 'New text'`        |
| `setAttribute(attribute, value)` | Sets the value of a specific attribute            | `element.setAttribute('src', 'image.jpg')` |
| `getAttribute(attribute)` | Gets the value of a specific attribute        | `let attr = element.getAttribute('src')` |
| `style`      |Used to change inline style property of an element    | `element.style.color = 'red'`  |

*   **`innerHTML`:**  It lets you set any HTML as content, including tags. So, it can be risky if you're using user input (it can lead to a security vulnerability called "Cross-Site Scripting" - XSS).
*   **`textContent`:** Sets the text content of the element, and it doesn't treat the text as HTML tags. Hence, it's safer.
*   **`setAttribute()`:** You can use this to set or modify the value of attributes.
*  **`getAttribute()`:** you can use this to get the value of attributes.
*  **`style`** is an object that represents inline style property of the element

**Example:**

```html
<!DOCTYPE html>
<html>
<head>
    <title>DOM Manipulation</title>
</head>
<body>
    <h1 id="my-heading">Original Heading</h1>
    <img id="my-image" src="image.png" alt="Old Image">
    <div id = "my-div" style="background-color:blue;">
      <p id = "my-paragraph">Original Text</p>
    </div>
    <script>
        let heading = document.getElementById('my-heading');
        let image = document.getElementById('my-image')
        let div = document.getElementById('my-div')
        let paragraph = document.getElementById('my-paragraph')

        heading.innerHTML = '<b>New Heading</b>'; // Change the HTML content
        image.setAttribute('src','new_image.png'); // change the src attribute
        image.setAttribute('alt','new_image')
        let srcValue = image.getAttribute('src')
        let altValue = image.getAttribute('alt')
        console.log("src: " + srcValue)
        console.log("alt: " + altValue)

        div.style.backgroundColor = "red"
        paragraph.textContent = "New paragraph"  // Change the text content
    </script>
</body>
</html>
```

**4. Adding and Removing HTML Elements**

*   **Creating Elements:** You can create new elements using the `document.createElement(tagName)` method

    ```javascript
    let newElement = document.createElement('p'); // Creates a new paragraph element
    ```
*  **Adding Elements:** To add the new element, we can use `appendChild()`, `insertBefore()` methods

   ```javascript
      let newElement = document.createElement('p'); // Creates a new paragraph element
      newElement.textContent = "This is a new paragraph"

      document.body.appendChild(newElement) // Add the paragraph as child of the body element

      let div = document.getElementById('my-div')
      let anotherElement = document.createElement('span')
      anotherElement.textContent = "This is new span"
      document.body.insertBefore(anotherElement, div); // Add span before div element

      div.appendChild(anotherElement) // Add the span element as a child of div element
  ```
*  **Removing Elements:**  You can remove an element using `removeChild()` or `remove()` methods
   ```javascript
        let div = document.getElementById('my-div')
        let heading = document.getElementById('my-heading')

        div.removeChild(heading); // Removes the heading element from the div container

        heading.remove(); // Removes the element from document
   ```
**Important Points:**

*   `appendChild` always appends the new element at last
*  `insertBefore` method requires the reference of the element before which the new element should be inserted.
*   When you remove an element, the child nodes are also removed

**5. Styling HTML Elements using JavaScript**

You can directly change the style of elements using the `style` property.

```javascript
let element = document.getElementById('my-div')
element.style.backgroundColor = 'red';
element.style.color = 'white';
element.style.padding = '10px';
```

*   Each CSS property becomes a property of the element's `style` object (in camelCase, e.g., `backgroundColor` instead of `background-color`).

**Example (from your instructions):**

*   The example mentioned in the instructions is using `document.getElementById()` to get a particular element and change its text using `innerHTML` or `textContent`. This is covered above.

**Expected Outcome:**

You should now be able to:

*   Understand the structure of the DOM tree.
*   Select HTML elements using `getElementById`, `getElementsByClassName`, `querySelector`, `querySelectorAll` etc.
*   Modify HTML elements using `innerHTML`, `textContent`, `setAttribute` etc.
*  Add and remove elements from the DOM
*   Style HTML elements using JavaScript.
*  Manipulate the webpage based on user behaviour.

That's it for Chapter 10, boss! DOM manipulation is essential for making dynamic web pages. Practice a lot and try out different things. Any questions, feel free to ask. Next, we will learn about Events! Let's go! 🚀
