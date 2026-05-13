---
name: Agentic RAG
colors:
  surface: '#051424'
  surface-dim: '#051424'
  surface-bright: '#2c3a4c'
  surface-container-lowest: '#010f1f'
  surface-container-low: '#0d1c2d'
  surface-container: '#122131'
  surface-container-high: '#1c2b3c'
  surface-container-highest: '#273647'
  on-surface: '#d4e4fa'
  on-surface-variant: '#bec8d2'
  inverse-surface: '#d4e4fa'
  inverse-on-surface: '#233143'
  outline: '#88929b'
  outline-variant: '#3e4850'
  surface-tint: '#89ceff'
  primary: '#89ceff'
  on-primary: '#00344d'
  primary-container: '#0ea5e9'
  on-primary-container: '#003751'
  inverse-primary: '#006591'
  secondary: '#c0c1ff'
  on-secondary: '#1000a9'
  secondary-container: '#3131c0'
  on-secondary-container: '#b0b2ff'
  tertiary: '#4ae176'
  on-tertiary: '#003915'
  tertiary-container: '#00b351'
  on-tertiary-container: '#003c16'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#c9e6ff'
  primary-fixed-dim: '#89ceff'
  on-primary-fixed: '#001e2f'
  on-primary-fixed-variant: '#004c6e'
  secondary-fixed: '#e1e0ff'
  secondary-fixed-dim: '#c0c1ff'
  on-secondary-fixed: '#07006c'
  on-secondary-fixed-variant: '#2f2ebe'
  tertiary-fixed: '#6bff8f'
  tertiary-fixed-dim: '#4ae176'
  on-tertiary-fixed: '#002109'
  on-tertiary-fixed-variant: '#005321'
  background: '#051424'
  on-background: '#d4e4fa'
  surface-variant: '#273647'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-mono:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.05em
  caption:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
  display-lg-mobile:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 36px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 8px
  gutter: 16px
  margin-mobile: 16px
  margin-desktop: 32px
  container-max: 1440px
---

## Brand & Style
The brand personality is rooted in **Precision, Intelligence, and Technical Authority**. As an enterprise-grade AI application, the UI must facilitate deep focus and instill trust through rigorous structural alignment and clarity.

The chosen design style is **Corporate / Modern** with a **Technical Minimalist** edge. This is achieved by pairing high-density data layouts with generous whitespace in information-heavy views. The aesthetic avoids unnecessary ornamentation, favoring functional elements like monospaced identifiers and status-driven accents to communicate the "Agentic" nature of the software—where the UI feels like a high-performance cockpit for AI orchestration.

Key emotional drivers:
- **Reliability:** Stable, dark-mode foundation for long-duration cognitive tasks.
- **Speed:** High-contrast accents (Electric Blue) to guide the eye to primary actions.
- **Transparency:** Clear visual distinction between AI-generated content and system metadata.

## Colors
This design system utilizes a sophisticated dark-mode palette designed to reduce ocular fatigue while maintaining high accessibility standards.

- **Foundations:** The core environment is built on a Deep Navy (#0f172a), providing a solid, ink-like depth. Surfaces use progressive Slate shades to indicate hierarchy without relying solely on shadows.
- **Accents:** Electric Blue serves as the primary driver for "Interactive" states and AI "Active" moments. Violet is reserved for secondary logic branches or "Agentic" process indicators.
- **Utility:** Success, Warning, and Error colors are slightly desaturated to harmonize with the dark background while remaining functionally distinct for status-dot indicators and critical alerts.
- **Text:** Near-white is reserved for core content to maximize legibility, while Muted Gray handles metadata, timestamps, and secondary labels to maintain clear information hierarchy.

## Typography
The typography strategy prioritizes readability and technical clarity. **Inter** is the workhorse for the system, providing a neutral yet modern tone that excels in digital interfaces.

**Typography Roles:**
- **Headlines:** Semi-bold to Bold weights with slight negative letter-spacing for a tight, professional appearance in dashboard titles and section headers.
- **Body:** Standardized at 14px and 16px for optimal balance between information density and legibility.
- **Monospace Accents:** **JetBrains Mono** is employed specifically for technical strings, such as Source IDs, File Paths, and RAG-retrieval metadata. This creates a "code-adjacent" feel that suits the AI-technical domain.
- **Labels:** Uppercase monospaced labels are used for technical status indicators to distinguish them from natural language.

## Layout & Spacing
The layout follows a **Fluid Grid** approach to accommodate the varying data-widths of AI chat interfaces and document side-panels.

**Grid Philosophy:**
- **12-Column System:** Used for top-level dashboard layouts. 
- **The 8px Rule:** All margins, paddings, and component heights are multiples of 8px to ensure a consistent rhythmic flow.
- **Contextual Containers:** Chat windows and document viewers utilize flexible max-widths to prevent line lengths from becoming unreadable on ultra-wide monitors.
- **Adaptive Strategy:** On mobile, margins shrink to 16px and the 12-column grid collapses into a single-column stack, prioritizing the "Conversation" or "Input" area.

## Elevation & Depth
Depth is communicated through **Tonal Layering** supplemented by **Subtle Borders** rather than aggressive shadows. This maintains the clean, enterprise feel.

- **Level 0 (Base):** #0f172a. The main application background.
- **Level 1 (Surfaces):** #1e293b. Used for sidebars, navigation rails, and grouped content areas. A 1px solid #334155 border defines the edge.
- **Level 2 (Cards/Modals):** #334155. Elevated elements used for tooltips, modals, and source-cards.
- **Shadows:** When necessary (e.g., floating action menus), use a soft, diffused shadow: `0 10px 15px -3px rgba(0, 0, 0, 0.4)`. The shadow color should be tinted with the base navy to feel integrated rather than "dirty."

## Shapes
The shape language is **Rounded**, striking a balance between modern friendliness and professional rigidity. 

- **Base Radius:** 8px (0.5rem) for standard components like buttons and input fields.
- **Large Radius:** 16px (1rem) for primary content containers and cards.
- **Pill Shapes:** Specifically reserved for "Badges" and "Source Chips" to distinguish them from interactive buttons.
- **Consistency:** Borders are consistently 1px. Avoid varied border-widths to maintain the technical, precise aesthetic.

## Components
- **Buttons:** Primary buttons use a solid Electric Blue fill with white text. Secondary buttons use an outlined style with the #334155 border and slate-white text.
- **Pill Badges:** Used for status (e.g., "Active," "Indexing"). They feature a low-opacity background of the status color with a high-contrast text label and a leading "status dot."
- **Source Chips:** Small, rounded-md containers with #334155 backgrounds, displaying a document icon and a monospaced ID. 
- **Input Fields:** Dark slate background with a 1px #334155 border. On focus, the border transitions to Electric Blue with a subtle outer glow.
- **Skeleton States:** Use a pulsing linear-gradient shimmer from #1e293b to #334155. Skeletons should mirror the exact border-radius of the content they represent.
- **Cards:** Defined by #334155 background and 1px border. Interactive cards should have a subtle hover state that lightens the background by 5%.