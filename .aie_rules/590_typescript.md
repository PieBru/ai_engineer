---
description: "Comprehensive TypeScript best practices for building robust, scalable, and maintainable applications."
author: "AI Engineer Team"
version: "1.0"
---

# TypeScript Best Practices

## Type System
- Prefer interfaces over types for object definitions and when extending (e.g., class implements interface).
- Use `type` for unions, intersections, mapped types, and when defining function signatures or tuple types.
- Avoid using `any` whenever possible; prefer `unknown` for truly unknown types and perform necessary type checking before use.
- Use `as const` for true immutability of object literals and arrays, and for creating enum-like structures with string or number literals.
- Utilize template literal types for more precise string typing (e.g., `type EventName = `${'click' | 'hover'}_${string}`;`).
- Leverage TypeScript's built-in utility types (e.g., `Partial<T>`, `Required<T>`, `Readonly<T>`, `Pick<T, K>`, `Omit<T, K>`, `Record<K, T>`, `ReturnType<F>`).
- Use generics (`<T>`) for reusable type patterns and to create type-safe functions, classes, and interfaces that can work over a variety of types.

## Naming Conventions
- Use PascalCase for type names, interfaces, enums, and classes (e.g., `UserProfile`, `StatusEnum`, `UserService`).
- Use camelCase for variables and functions (e.g., `userName`, `calculateTotal`).
- Use UPPER_CASE for constants (e.g., `MAX_USERS`, `API_ENDPOINT`). Enum members are often PascalCase by convention (e.g., `StatusEnum.Active`).
- Use descriptive names with auxiliary verbs (e.g., `isLoading`, `hasError`, `canSubmit`) for boolean variables and functions returning booleans.
- Prefix interfaces for React props with 'Props' (e.g., `ButtonProps`) if that's the established project convention.
- Handle unused parameters explicitly: if `noUnusedParameters` is enabled in `tsconfig.json`, prefix unused parameters with an underscore (e.g., `_unusedParam`) to indicate they are intentionally unused, or remove them if truly not needed.

## Code Organization
- Keep type definitions close to where they're used for types local to a module or component.
- Export types and interfaces from dedicated type files (e.g., `types.ts`, `interfaces.ts`, or domain-specific files like `user.types.ts`) when they are shared across multiple modules.
- Use barrel exports (`index.ts`) judiciously for organizing exports from a directory, but be mindful of potential circular dependencies and bundle size impacts.
- Place broadly shared types in a `types` or `interfaces` directory at an appropriate scope (e.g., `src/types`).
- Configure and use module path aliases in `tsconfig.json` (e.g., `"paths": { "@/*": ["src/*"] }`) for cleaner imports in larger projects (e.g., `import MyComponent from '@/components/MyComponent';`).
- Co-locate component props with their components (common in UI development, e.g., defining `MyComponentProps` in `MyComponent.tsx`).

## Functions
- Use explicit return types for all public functions and methods to ensure clarity and prevent accidental type changes.
- Use arrow functions for callbacks and class methods where `this` context might be an issue or for concise syntax.
- Implement proper error handling. Consider using custom error types (e.g., `class ValidationError extends Error {}`) or clearly defining expected error shapes for API responses.
- Use function overloads for complex type scenarios where a function can accept different argument types and/or return different types based on input.
- Prefer `async/await` over raw `Promise.then().catch()` chains for cleaner and more readable asynchronous code.
- Ensure `async` functions always return a `Promise` (e.g., `async function fetchData(): Promise<DataType>` or `async function doSomething(): Promise<void>` if no value is returned).

## Best Practices
- Enable `strict` mode in `tsconfig.json` (this enables a suite of strict type-checking options).
- Use `readonly` for properties that should not be reassigned after object creation, and `Readonly<T>` or `ReadonlyArray<T>` for immutable collections.
- Leverage discriminated unions (tagged unions) for type safety when dealing with objects that can be one of several shapes.
- Use type guards (e.g., `typeof`, `instanceof`, custom functions returning `value is Type`) for runtime type checking and narrowing types within conditional blocks.
- Implement proper null and undefined checking, especially when `strictNullChecks` is enabled. Use optional chaining (`?.`) and nullish coalescing (`??`).
- Avoid type assertions (`value as Type` or `<Type>value`) unless absolutely necessary and you are certain about the type. Prefer type guards or conditional checks to narrow types safely.
- Use the `satisfies` operator (TypeScript 4.9+) to validate that an expression matches some type, without changing the actual inferred type of that expression.
- Consider enabling `noUncheckedIndexedAccess` in `tsconfig.json` for safer object and array property access, as it adds `| undefined` to the type of indexed access.

## Error Handling
- Create custom error types by extending `Error` for domain-specific errors (e.g., `class NetworkError extends Error { constructor(message: string, public statusCode?: number) { super(message); } }`).
- Consider using Result types (e.g., from libraries like `neverthrow`, `oxide.ts`, or a custom `type Result<T, E> = { success: true; value: T } | { success: false; error: E };`) for operations that can fail predictably, as an alternative to throwing exceptions for expected errors.
- Implement proper error boundaries (especially in UI frameworks like React) to catch errors in component subtrees and display fallback UI.
- Use `try...catch` blocks with typed catch clauses (`catch (e: unknown)` and then perform type checking on `e`, e.g., `if (e instanceof MyCustomError)`).
- Handle Promise rejections properly with `.catch()` or `try/catch` with `await`. Ensure all promises are handled to avoid unhandled promise rejections.
- Consider using a dedicated error reporting service (e.g., Sentry, Bugsnag) in production environments to track and analyze errors.

## Patterns
- Use the Builder pattern for constructing complex objects step-by-step, especially when objects have many optional properties.
- Implement the Repository pattern for abstracting data access logic, providing a clean API for data operations.
- Use the Factory pattern for object creation when the exact type isn't known until runtime or to encapsulate creation logic.
- Leverage dependency injection (DI) for decoupling components and improving testability. DI can be manual or facilitated by libraries (e.g., `tsyringe`, `InversifyJS`).
- Use ES modules (`import`/`export`) for encapsulation and organizing code into reusable units.
- Prefer functional programming paradigms (e.g., immutability, pure functions, higher-order functions) where they enhance clarity, maintainability, and testability.

## `tsconfig.json` Configuration
- Maintain a well-configured `tsconfig.json` at the root of your project.
- Enable `strict` mode. This enables a suite of strictness flags:
  - `noImplicitAny`: Raise error on expressions and declarations with an implied `any` type.
  - `strictNullChecks`: When true, `null` and `undefined` have their own distinct types and you'll get errors where you could be using a value that might be `null` or `undefined`.
  - `strictFunctionTypes`: Enables stricter checking of function type parameters.
  - `strictBindCallApply`: Enables stricter checking of `bind`, `call`, and `apply` methods on functions.
  - `strictPropertyInitialization`: Ensures class properties are initialized in the constructor or by a property initializer.
  - `noImplicitThis`: Raise error on `this` expressions with an implied `any` type.
  - `alwaysStrict`: Parse in strict mode and emit "use strict" for each source file.
- Use `esModuleInterop: true` for better compatibility between CommonJS and ES modules.
- Use `resolveJsonModule: true` to allow importing `.json` files.
- Configure `target` (e.g., `ES2020`, `ES2021`, `ESNext`) and `module` (e.g., `ESNext`, `NodeNext`, `CommonJS`) appropriately for your runtime environment and bundler.
- Set `moduleResolution` to `bundler` (TypeScript 5.0+), `node16`, or `nodenext` for modern Node.js/bundler module resolution behavior.
- Consider `skipLibCheck: true` to speed up type checking by skipping type checking of declaration files (`.d.ts`), but be aware of the implications.

## Tooling
- Use ESLint with TypeScript-specific plugins (e.g., `@typescript-eslint/eslint-plugin` and `@typescript-eslint/parser`) for comprehensive linting beyond what `tsc` provides.
- Use Prettier for consistent code formatting, ideally with a TypeScript plugin or integrated with ESLint (`eslint-config-prettier`).
- Integrate linters and formatters into pre-commit hooks (e.g., using Husky and lint-staged) to maintain code quality automatically.

## Interoperability with JavaScript
- Provide declaration files (`.d.ts`) for JavaScript libraries if they are not available from `@types` packages on npm (e.g., `@types/lodash`).
- Gradually migrate JavaScript codebases to TypeScript where beneficial. Use `allowJs: true` and `checkJs: true` options in `tsconfig.json` during the transition to allow TypeScript to check JavaScript files and to import JS into TS.

## Testing with TypeScript
- Write tests in TypeScript using frameworks like Jest, Vitest, Mocha, Playwright, or Cypress.
- Utilize TypeScript's type system for creating typed mocks, test data, and for ensuring test setup and assertions are type-safe, improving test reliability and maintainability.
