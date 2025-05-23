Okay boss, let's dive into Chapter 13: Working with JSON. JSON is a very important data format for transferring data, especially when working with APIs and web services. We'll cover everything from understanding JSON to parsing and stringifying it, ensuring you become a JSON expert!

**Chapter 13: Working with JSON**

**1. What is JSON?**

*   **JSON (JavaScript Object Notation):** It's a lightweight, text-based data format that's easy for humans to read and write, and for machines to parse and generate.
*   **Key Features:**
    *   It's based on a subset of JavaScript syntax, but it's language-independent.
    *   Data is represented in key-value pairs.
    *   It's used to transmit data in web applications (between client and server).
*   **Data Types in JSON:**
    *   **Strings:** Enclosed in double quotes (`"`) e.g., `"hello"`
    *   **Numbers:** Integers or floating-point numbers e.g., `10`, `3.14`
    *   **Booleans:** `true` or `false`
    *   **Null:** `null`
    *   **Objects:** Key-value pairs enclosed in curly braces `{}` e.g., `{"name": "Raju", "age": 30}`
    *   **Arrays:** Ordered lists enclosed in square brackets `[]` e.g., `[1, 2, 3]`, `["apple", "banana"]`

**Example JSON data:**

```json
{
  "name": "John Doe",
  "age": 30,
  "city": "New York",
  "isStudent": false,
  "hobbies": ["reading", "coding", "hiking"],
    "address": {
        "street": "123 Main St",
        "zip": "10001"
    }
}
```

**Key differences between JSON and Javascript objects:**

|Feature| Javascript Object | JSON|
|---|---|---|
|Keys| Key can be string, numbers, Symbols| Keys must be string in double quotes |
|Values| Any Javascript data type | Strings, Numbers, Booleans, null, objects and arrays |
|Syntax| Can be created using both object literals and constructors| Can only be created with object literals. There will not be any function inside json.|

**2. Parsing JSON Data (`JSON.parse()`)**

*   **Parsing:** Converting a JSON string into a JavaScript object (or array).
*   **`JSON.parse(jsonString)`:** This method takes a JSON string as an argument and returns the corresponding JavaScript object.

```javascript
let jsonString =
`{
  "name": "John Doe",
  "age": 30,
  "city": "New York"
}`;

let jsObject = JSON.parse(jsonString);
console.log(jsObject); // Output: { name: 'John Doe', age: 30, city: 'New York' }
console.log(jsObject.name) // Output: John Doe
console.log(jsObject.age) // Output: 30
```

*   The result of `JSON.parse` is a JavaScript object, so you can access the properties using dot or bracket notation.
*  If the input is not a valid JSON string, then it will throw an error.

**3. Stringifying JavaScript Objects (`JSON.stringify()`)**

*   **Stringifying:** Converting a JavaScript object into a JSON string.
*   **`JSON.stringify(jsObject)`:** This method takes a JavaScript object (or array) as an argument and returns the corresponding JSON string.

```javascript
let person = {
  name: "Jane Smith",
  age: 25,
  city: "London",
};

let jsonString = JSON.stringify(person);
console.log(jsonString); // Output: {"name":"Jane Smith","age":25,"city":"London"}
console.log(typeof jsonString); // Output: string
```

*   The result of `JSON.stringify` is a JSON string.
*   This JSON string can be used to transfer data.

**4. Fetching JSON Data using API Call**

*   APIs (Application Programming Interfaces) often return data in JSON format.
*   You can use the `fetch()` function to make an HTTP request to an API and retrieve the JSON data.

```javascript
async function fetchData() {
  try {
    const response = await fetch('https://jsonplaceholder.typicode.com/todos/1');
    const jsonData = await response.json(); //Parse the response as JSON
    console.log(jsonData);
  } catch (error) {
    console.error('Error fetching data:', error);
  }
}
fetchData()
```

*   The `fetch()` function returns a promise, which needs to be resolved to get the actual data.
*   The `response.json()` is another promise, which will parse the response body as JSON.

**5. Example: JSON Data Fetching and Rendering**

Let's combine everything to fetch JSON data from an API and display it on the web page:

```html
<!DOCTYPE html>
<html>
<head>
    <title>JSON Data</title>
</head>
<body>
    <div id="data-container"></div>
    <script>
        async function fetchDataAndRender() {
            try {
                const response = await fetch('https://jsonplaceholder.typicode.com/todos/1');
                const jsonData = await response.json();

                let container = document.getElementById('data-container');
                container.innerHTML = `
                    <h2>${jsonData.title}</h2>
                    <p>User ID: ${jsonData.userId}</p>
                    <p>Completed: ${jsonData.completed}</p>
                `;
            } catch (error) {
                let container = document.getElementById('data-container');
                container.innerHTML = `<p>Error fetching data: ${error}</p>`;
                console.error("Error: ", error)
            }
        }
        fetchDataAndRender()
    </script>
</body>
</html>
```

*   Here we are fetching data from the api.
*   When the data is fetched, we create new elements and set the data to it.
*   This example fetches data from an API and displays a title, userID and completed status in the HTML document.

**Expected Outcome:**

You should now be able to:

*   Understand what JSON is and its data types.
*   Convert JSON strings to JavaScript objects using `JSON.parse()`.
*   Convert JavaScript objects to JSON strings using `JSON.stringify()`.
*   Fetch JSON data from an API using `fetch()`.
*  Parse the JSON response and render the data in the UI.

That’s all for Chapter 13, boss! JSON is a very important data format you will deal with in many web application. Practice with different JSON data, parsing and stringifying to become comfortable. Any doubts? Just ask! We are heading to error handling next! Let's go! 🚀
