# Archived Frontend Components (Dosing & Optimizers)

This archive folder contains frontend dosing recommender and simulator components that have been temporarily removed from active views:

### 1. `DosingOptimizerCard.tsx`
- **Original Location**: `src/pages/PoolDetailPage.tsx`
- **Description**: Displays automated hypochlorite dosing pump power (%), recommended pump recirculation time (hours), and projected chlorine levels based on the backend `OptimiserResult` (`/optimise/{pool_id}`).

### 2. `DosingSimulator.tsx`
- **Original Location**: `src/pages/CleaningPage.tsx`
- **Description**: Interactive pool volume, current/target chlorine, and dosing pump rate simulator for calculating chemical injection hours and required liters of hypochlorite.

### 3. `DosingPreferenceToggle.tsx`
- **Original Location**: `src/pages/AccountPage.tsx`
- **Description**: Configuration preference toggle switch for automated dosing calculations.

---

### How to Restore

Import any of the components from `../archive` or copy their JSX directly back into their respective pages:
```tsx
import { DosingOptimizerCard } from "../archive";
// or
import { DosingSimulator } from "../archive";
```
