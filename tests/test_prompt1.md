You are the best AI Coding Engineer.

I need your help to create an interactive web animation. Please write the complete code for a single HTML file that displays a ball bouncing realistically inside a continuously spinning hexagon.

Here are the specific requirements for this animation:

1.  **Core Technology:**
    *   The entire application (HTML structure, CSS styling, and JavaScript logic) must be contained within a single `.html` file.

2.  **Visual Elements & Styling (CSS):**
    *   **Page:** The main `<body>` should have an engaging background, perhaps a linear gradient. Center the content on the page.
    *   **Title:** Display a clear title for the animation, like "Ball in Spinning Hexagon," styled for visibility (e.g., white text with a subtle shadow).
    *   **Canvas:**
        *   Use an HTML5 `<canvas>` element for rendering the animation. A good size would be around 600x600 pixels.
        *   The canvas itself should have a background color (e.g., a semi-transparent dark color) and rounded corners with a slight box-shadow to make it stand out.
    *   **Ball:** The ball should be a circle with a radius of about 15 pixels. Give it a vibrant color and apply a radial gradient to create a simple 3D effect.
    *   **Hexagon:** The hexagon should be drawn as an outline (stroke) with a contrasting color. It should be large enough (e.g., radius of 200 pixels) to contain the bouncing ball and centered on the canvas.

3.  **Animation & Physics (JavaScript):**
    *   **Ball Dynamics:**
        *   The ball should start with an initial velocity.
        *   Implement gravity that pulls the ball downwards.
        *   Apply friction (or damping) to the ball's movement and especially upon bouncing, so it doesn't bounce indefinitely with the same energy.
    *   **Hexagon Dynamics:**
        *   The hexagon should rotate smoothly around its center.
    *   **Collision & Bounce:**
        *   Implement accurate collision detection between the ball and the six sides of the spinning hexagon.
        *   When a collision occurs, the ball should bounce off the wall realistically (angle of incidence/reflection, adjusted for wall's current angle due to rotation).
    *   **Animation Loop:** Use `requestAnimationFrame` for smooth and efficient animation.

4.  **User Controls (HTML & JavaScript):**
    *   Below the canvas, include a "controls" section.
    *   Add three range input sliders:
        *   One for adjusting the strength of **Gravity** (e.g., min 0, max 1, step 0.01, default 0.2).
        *   One for adjusting **Friction** (e.g., min 0, max 0.1, step 0.001, default 0.01).
        *   One for adjusting the **Rotation Speed** of the hexagon (e.g., min 0, max 0.1, step 0.001, default 0.01).
    *   Each slider should have a clear label.
    *   The JavaScript should update the corresponding physics parameters in real-time as the user interacts with the sliders.

5.  **JavaScript Code Structure:**
    *   Organize the JavaScript code logically. Consider using functions for:
        *   Initialization of canvas and variables.
        *   Drawing the hexagon (considering its current rotation).
        *   Drawing the ball.
        *   Updating the physics state (applying gravity, friction, updating positions).
        *   Checking for and handling collisions.
        *   The main animation loop function.
        *   Event handlers for the sliders.

Please provide the complete HTML code as a single block. Ensure the code is well-structured and includes comments where helpful for understanding the logic.
