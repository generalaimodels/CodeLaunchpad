function cancellableTimeout(ms) {
  let cancelled = false;
  const promise = new Promise((res, rej) => {
    const id = setTimeout(() => cancelled ? rej('Cancelled') : res('Done'), ms);
    if (cancelled) clearTimeout(id);
  });
  return {
    promise,
    cancel: () => { cancelled = true; }
  };
}
const c1 = cancellableTimeout(100);
c1.cancel();
c1.promise.catch(console.error);