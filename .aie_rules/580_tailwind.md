---
description: "Comprehensive Tailwind CSS best practices for building modern, scalable, and maintainable user interfaces."
author: "AI Engineer Team"
version: "1.0"
---

# Tailwind CSS Best Practices

## Project Setup
- Use proper Tailwind configuration
- Configure theme extension properly
- Set up proper content configuration (formerly purge) to ensure all used classes are included in the build.
- Use proper plugin integration
- Configure custom spacing and breakpoints
- Set up proper color palette
- Integrate with PostCSS for additional processing if needed (e.g., autoprefixer, nesting).
- Use a `.gitignore` file to exclude build artifacts (e.g., `dist/`, `public/build/`).

## Component Styling
- Use utility classes over custom CSS
- Group related utilities with @apply when needed
- Use proper responsive design utilities
- Implement dark mode properly
- Use proper state variants
- Use group and peer variants for styling based on parent or sibling state.
- Keep component styles consistent

## Layout
- Use Flexbox and Grid utilities effectively
- Implement proper spacing system
- Use container queries when needed
- Implement proper responsive breakpoints
- Use proper padding and margin utilities
- Implement proper alignment utilities
- Utilize aspect ratio utilities for maintaining dimensions.

## Typography
- Use proper font size utilities
- Implement proper line height
- Use proper font weight utilities
- Configure custom fonts properly
- Use proper text alignment
- Implement proper text decoration
- Use text overflow utilities for handling long text.

## Colors
- Use semantic color naming
- Implement proper color contrast
- Use opacity utilities effectively
- Configure custom colors properly
- Use proper gradient utilities
- Implement proper hover states

## Components
- Use shadcn/ui components when available
- Extend components properly
- Keep component variants consistent
- Implement proper animations
- Use proper transition utilities
- Ensure custom components are designed with Tailwind's utility-first approach in mind.
- Keep accessibility in mind

## Responsive Design
- Use mobile-first approach
- Implement proper breakpoints
- Use container queries effectively
- Handle different screen sizes properly
- Implement proper responsive typography
- Use proper responsive spacing
- Test responsiveness thoroughly across various devices and viewports.

## Performance
- Use proper content configuration (formerly purge) to minimize final CSS bundle size.
- Minimize custom CSS
- Use proper caching strategies
- Implement proper code splitting
- Optimize for production
- Monitor bundle size
- Consider using JIT (Just-In-Time) mode (default in Tailwind CSS v3+) for faster builds and smaller CSS.

## Best Practices
- Follow naming conventions
- Keep styles organized
- Use proper documentation
- Implement proper testing
- Use linters (e.g., ESLint with Tailwind CSS plugins) and formatters (e.g., Prettier with `prettier-plugin-tailwindcss`) for code consistency.
- Implement pre-commit hooks (e.g., Husky with lint-staged) to enforce styling and linting.
- Follow accessibility guidelines
- Use proper version control 

## Advanced Configuration
- Utilize Tailwind's theming capabilities to customize design tokens.
- Create custom utility classes or components with `@layer` directives if necessary, but prefer utilities.
- Understand how to configure variants for utilities (e.g., `responsive`, `hover`, `focus`).
- Use `tailwind.config.js` (or `.ts`) effectively to manage all customizations.
