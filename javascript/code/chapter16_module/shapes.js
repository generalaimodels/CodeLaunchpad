// This module defines classes for different shapes.

export class Circle { // ⚪ Class for Circle
    constructor(radius) {
        this.radius = radius;
    }
    getArea() {
        return PI * this.radius * this.radius; // Using PI from same module or another imported module (if exported)
    }
}

export class Square { // ⬜ Class for Square
    constructor(side) {
        this.side = side;
    }
    getArea() {
        return this.side * this.side;
    }
}
