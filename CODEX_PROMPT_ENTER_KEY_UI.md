# Codex Prompt: Enter Key & Professional UI Enhancement

## Objective
Enhance the FastAPI Agentic RAG web interface to support automatic message sending on Enter key press and elevate the UI design to professional-grade standards with modern aesthetics, improved spacing, and refined visual hierarchy.

## Requirements

### 1. Enter Key Functionality
- **Default Behavior**: When user presses **Enter** key in the message textarea, the form should automatically submit and send the message.
- **Shift+Enter Behavior**: Pressing **Shift+Enter** should create a new line in the textarea (preserve existing newline functionality).
- **Implementation Location**: Update the JavaScript event handler for `#question` textarea in the browser application.
- **User Experience**: No visible button press required—pressing Enter immediately sends the message.

### 2. Professional UI Design Enhancement
Upgrade the existing dark blue theme with modern design principles:

#### Typography & Spacing
- Increase font sizes for better readability: headings +2-4px, body text +1px
- Improve line-height for better readability (1.6–1.8 for body copy)
- Add more breathing room with increased padding (14–16px in cards vs. current 12px)
- Use better font weight hierarchy (700 for labels, 600 for headings)

#### Visual Hierarchy & Cards
- Add subtle gradients or layered backgrounds to key panels
- Increase box-shadow depth for better depth perception
- Add hover effects on interactive elements (buttons, cards, session cards)
- Improve color contrast ratios for WCAG accessibility

#### Color & Accents
- Enhance the cyan/teal accent colors with more vibrant states
- Add color-coded badges for message types (user=blue, assistant=teal/cyan)
- Improve the message bubble styling with better borders and backgrounds
- Add gradient underlines or accent borders to important sections

#### Component Refinement
- **Message Bubbles**: Round corners more, add subtle shadows, improve spacing
- **Buttons**: Add hover/active states, rounded corners (current 8px is good), better focus states
- **Input Fields**: Slightly larger padding, better focus ring styling
- **Badges**: More polished appearance with gradient accents
- **Panels**: Add subtle top border accent color

#### Layout Improvements
- Better alignment and spacing in the composer section
- Improved padding consistency across all panels
- Enhanced visual separation between sections
- Better responsive design transitions

#### Animation & Interactivity
- Add smooth transitions on hover (0.15s–0.2s)
- Button press feedback (slight scale or color shift)
- Smooth message appearance animation
- Loading state indicator improvements

### 3. Specific Code Changes

#### JavaScript Changes
```javascript
// In the message form event listener section:
// Add handler for Ctrl/Cmd+Enter or just Enter in textarea
$("#question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#askForm").dispatchEvent(new Event("submit"));
  }
  // Shift+Enter creates new line (default behavior continues)
});
```

#### CSS Enhancements
- Increase `.message` padding from `13px 14px` to `14px 16px`
- Increase `.panel.pad` padding from `18px` to `20px`
- Add hover effects: `transition: all 0.15s ease;`
- Enhance `.message-row.user .message` with a gradient overlay
- Improve `.badge` styling with semi-transparent colored borders
- Add `:hover` and `:active` states for `.session-card`, buttons
- Update `.composer textarea` to `min-height: 120px` for better proportions

### 4. Deliverables
- Modified `app.py` with:
  - Updated HTML/CSS with professional styling enhancements
  - JavaScript event handler for Enter key message submission
  - Improved visual design throughout the interface
- All existing functionality preserved (RAG chat, citations, guardrails, HITL)
- Responsive design maintained for mobile/tablet views
- No breaking changes to the `/api/chat` endpoint or data models

### 5. Testing Checklist
- [ ] Pressing Enter in message textarea sends the message
- [ ] Shift+Enter creates a new line in the textarea
- [ ] All buttons have visible hover/active states
- [ ] Message bubbles display with improved styling
- [ ] Badges and status indicators are clearly visible
- [ ] Layout looks polished on desktop and mobile views
- [ ] Existing RAG functionality (citations, guardrails) still works
- [ ] HITL checkpoints and approvals still functional

---

## Notes
This prompt is designed to be passed directly to GitHub Copilot or other code-generation models for implementation in `app.py`. The changes focus on UX improvement and professional visual design without altering backend logic or API contracts.
