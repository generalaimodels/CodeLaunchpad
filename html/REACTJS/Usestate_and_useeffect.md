# React Hooks: useState & useEffect

```javascript
import { useState, useEffect } from "react";
```

---

## 1. Overview  
React Hooks enable function components to manage state and side-effects.  
- `useState` → Local component state  
- `useEffect` → Side-effect management (data fetch, subscriptions, DOM side-effects)

---

## 2. useState Hook  

### 2.1 Purpose  
- Store and update primitive or complex state inside functional components  
- Trigger re-render on state change  

### 2.2 Syntax  
```js
const [state, setState] = useState(initialValue);
```

### 2.3 Initial State  
- **Primitive**: number, string, boolean.  
- **Object/Array**: full structure or factory function.  
- **Lazy initialization**:  
  ```js
  const [count, setCount] = useState(() => expensiveComputation());
  ```

### 2.4 State Updater  
- Calling `setState(newValue)` schedules a re-render.  
- **Replacement**: If state is object/array, updater replaces, not merges.  
- **Functional update**:  
  ```js
  setCount(prev => prev + 1);
  ```

### 2.5 Batching Behavior  
- Multiple `setState` calls within React event handlers batch before final render.  
- Outside React (e.g. native event), may not batch automatically.

### 2.6 Common Pitfalls  
- **Stale closures**: referencing old state in asynchronous callbacks.  
- **Object merging**: forgetting to copy spread when updating nested fields.  

---

## 3. useEffect Hook  

### 3.1 Purpose  
- Perform side-effects: data fetching, subscriptions, manual DOM updates, timers.

### 3.2 Syntax  
```js
useEffect(() => {
  // effect logic
  return () => {
    // cleanup logic
  };
}, [dep1, dep2]);
```

### 3.3 Dependency Array  
- **Empty array `[]`** → run once after mount (componentDidMount).  
- **Omitted** → run after every render.  
- **Specific dependencies** → run when any dependency’s reference changes.

### 3.4 Cleanup Function  
- Returned function executes before next effect or on unmount.  
- Use for unsubscribing, clearing timers, aborting fetch.

### 3.5 Timing & Order  
1. Render phase  
2. Commit phase: DOM updates  
3. Execute layout effects (`useLayoutEffect`)  
4. Execute normal effects (`useEffect`)

### 3.6 Common Patterns  
- **Data fetching**:  
  ```js
  useEffect(() => {
    let canceled = false;
    fetch(url)
      .then(res => res.json())
      .then(data => !canceled && setData(data));
    return () => { canceled = true; };
  }, [url]);
  ```
- **Event listeners**:  
  ```js
  useEffect(() => {
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  ```
- **Subscription management**: WebSocket, RxJS, etc.

### 3.7 Common Pitfalls  
- **Missing dependencies** → stale values or repeated effects.  
- **Over-subscription** → duplicate listeners on every render.  
- **Blocking renders** with heavy synchronous logic.

---

## 4. Combined Usage Patterns  
```jsx
function Profile({ userId }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/users/${userId}`)
      .then(res => res.json())
      .then(data => {
        setUser(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [userId]);

  if (loading) return <Spinner />;
  return <UserCard user={user} />;
}
```

- **State** tracks UI data.  
- **Effect** synchronizes component with external resources.  

---

## 5. Best Practices  

- Always declare all dependencies in the dependency array.  
- Use multiple `useEffect` calls to separate concerns.  
- Memoize handlers (`useCallback`) if passed to optimized children.  
- Prefer functional state updates when new state depends on previous.  
- Keep effect logic minimal; extract heavy computations outside or into custom hooks.

---

**Pros of Hooks**  
- Eliminates class boilerplate  
- Clear separation of concerns  
- Reusable logic via custom hooks  

**Cons / Considerations**  
- Learning curve: closures & dependencies  
- Overuse can lead to fragmented logic if not organized properly  

---

*End of Topic*