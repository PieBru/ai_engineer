---
description: "Comprehensive Svelte and SvelteKit best practices for building modern, performant, and maintainable web applications."
author: "AI Engineer Team"
version: "1.0"
---

# Svelte Best Practices

## Component Structure
- Keep components small and focused
- Name components with PascalCase (e.g., `MyComponent.svelte`).
- Use proper TypeScript integration
- Implement proper props typing (use `export let` for props)
- Use proper event dispatching (`createEventDispatcher`)
- Keep markup clean and readable
- Use `$$slots` to check for passed slots and conditionally render.
- Use proper slot implementation

## Reactivity
- Use proper reactive declarations
- Implement proper stores
- Use proper reactive statements (`$:`)
- Handle derived values properly
- Use proper lifecycle functions (`onMount`, `onDestroy`, `beforeUpdate`, `afterUpdate`)
- Use `tick()` when you need to wait for the DOM to update after a state change.
- Implement proper bindings (`bind:value`, `bind:this`, etc.)

## State Management
- Use proper Svelte stores
- Keep stores modular
- Use proper derived stores
- Implement proper actions
- Handle async state properly
- Unsubscribe from stores in `onDestroy` if manually subscribed, or use auto-subscription (`$storeName`).
- Use proper store subscriptions

## Performance
- Use proper component lazy loading
- Implement proper transitions
- Use proper animations
- Avoid unnecessary reactivity
- Use proper event forwarding
- Use `{#key ...}` blocks to manage component lifecycles when data changes.
- Implement proper key blocks

## Routing
- Use SvelteKit for routing
- Implement proper layouts
- Use proper route parameters
- Handle loading states properly
- Implement proper error pages
- Utilize SvelteKit's `load` functions for data fetching in routes.
- Use SvelteKit's form actions for handling form submissions.
- Use proper navigation methods

## Forms
- Use proper form bindings
- Implement proper validation
- Handle form submission properly
- Show proper loading states
- Use proper error handling
- Use progressive enhancement with SvelteKit form actions.
- Implement proper form reset

## TypeScript Integration
- Use proper component types
- Implement proper prop types
- Use proper event types
- Handle proper type inference
- Use proper store types
- Use `svelte-check` for validating Svelte files with TypeScript.
- Implement proper action types

## Testing
- Write proper unit tests
- Implement proper component tests
- Use proper testing libraries
- Test stores properly
- Use testing libraries like Vitest, Jest, and Svelte Testing Library.
- Implement proper mocking
- Test async operations

## Best Practices
- Follow Svelte style guide
- Use proper naming conventions
- Keep components organized
- Implement proper error handling
- Use proper event handling
- Use a `.gitignore` file to exclude `node_modules`, `.svelte-kit`, `build`, etc.
- Manage environment variables using SvelteKit's built-in mechanisms (e.g., `$env/static/public`, `$env/dynamic/private`).
- Document complex logic

## Build and Tooling
- Use Vite for development
- Configure proper build setup
- Use proper environment variables
- Use linters (e.g., ESLint with Svelte plugins) and formatters (e.g., Prettier with `prettier-plugin-svelte`).
- Implement pre-commit hooks (e.g., Husky with lint-staged).
- Implement proper code splitting
- Use proper asset handling
- Configure proper optimization 

## Accessibility (A11y)
- Use semantic HTML.
- Ensure keyboard navigability.
- Use ARIA attributes where necessary.
- Test with screen readers.
- Provide sufficient color contrast.
- Add `alt` text for images.
