# Feature Flags System

## Overview
This application uses a **Feature Flags** system to control which pages and features are visible to users. This is similar to how apps like Zepto toggle features like "Cafe" or "Grocery" on their homepage.

## How It Works

### 1. Configuration File (`feature_flags.json`)
All features are controlled through a single JSON file:

```json
{
  "features": {
    "settings": {
      "enabled": false,  // Set to true/false to enable/disable
      "name": "Settings",
      "description": "User settings and preferences"
    }
  }
}
```

### 2. Feature Flags Manager (`utils/feature_flags.py`)
Centralized system that checks if features are enabled:

```python
from utils.feature_flags import feature_flags

# Check if a feature is enabled
if feature_flags.is_enabled('settings'):
    # Show the feature
    pass
else:
    # Hide the feature
    st.warning("Feature disabled")
    st.stop()
```

### 3. Admin Dashboard (Page 11_Feature_Management.py)
**Admin users** can toggle features on/off through a user-friendly interface:
- Login as `admin`
- Navigate to "Feature Management" page
- Toggle features on/off with switches
- Click "Save Changes"

## Quick Start Guide

### To Hide Settings Page:
1. Open `feature_flags.json`
2. Find `"settings"` section
3. Change `"enabled": true` to `"enabled": false`
4. Save the file
5. Refresh your browser

**OR use the Admin Dashboard:**
1. Login as `admin`
2. Go to "Feature Management" page
3. Toggle "Settings" switch to OFF
4. Click "Save Changes"

### To Add a New Feature Flag:
1. Open `feature_flags.json`
2. Add a new entry:
```json
"my_new_feature": {
  "enabled": true,
  "name": "My New Feature",
  "description": "What this feature does"
}
```

3. In your page file, add the check:
```python
from utils.feature_flags import feature_flags

if not feature_flags.is_enabled('my_new_feature'):
    st.warning("This feature is currently disabled")
    st.stop()
```

## Available Features

| Feature Key | Page Name | Default State |
|------------|-----------|---------------|
| `upload_data` | Upload Data | Enabled |
| `model_selection` | Model Selection | Enabled |
| `dashboard` | Dashboard | Enabled |
| `reports` | Reports | Enabled |
| `settings` | Settings | **Disabled** |
| `demand_forecasting` | Demand Forecasting | Enabled |
| `crm_analytics` | CRM Analytics | Enabled |
| `facility_layout` | Facility Layout | Enabled |
| `ai_chatbot` | AI Chatbot | Enabled |
| `store_location_gis` | Store Location GIS | Enabled |

## Use Cases

### 1. **Temporarily Disable Features**
Turn off features during maintenance or when bugs are found:
```json
"ai_chatbot": {
  "enabled": false  // Users won't see AI Chatbot page
}
```

### 2. **Gradual Rollout**
Enable features for testing before full release:
```json
"new_analytics": {
  "enabled": false  // Turn on when ready
}
```

### 3. **A/B Testing**
Show different features to different users (requires additional logic)

### 4. **Emergency Disable**
Quickly disable problematic features without code deployment

## Admin Access

**Default Admin Account:**
- Username: `admin`
- Create account using `init_admin.py` script

**Admin Capabilities:**
- View all feature flags
- Toggle features on/off
- Save changes instantly
- See feature status overview

## Technical Details

### Singleton Pattern
The Feature Flags Manager uses a singleton pattern, ensuring one instance across the entire application.

### File Location
- Config: `feature_flags.json` (root directory)
- Manager: `utils/feature_flags.py`
- Admin Page: `pages/11_Feature_Management.py`

### Methods Available

```python
from utils.feature_flags import feature_flags

# Check if enabled
feature_flags.is_enabled('feature_key')  # Returns bool

# Get all enabled features
feature_flags.get_enabled_features()  # Returns dict

# Get all features (enabled + disabled)
feature_flags.get_all_features()  # Returns dict

# Runtime changes (not persisted)
feature_flags.enable_feature('feature_key')
feature_flags.disable_feature('feature_key')
```

## Best Practices

1. **Always check features at the page level** - Add checks at the top of each page
2. **Use descriptive names** - Make it clear what each feature does
3. **Document changes** - Keep track of why features are disabled
4. **Test before disabling** - Ensure no dependencies break
5. **Use Admin Dashboard** - Easier than manual JSON editing

## Troubleshooting

**Feature not hiding?**
- Clear browser cache
- Check if feature key matches exactly
- Verify JSON syntax is correct
- Restart Streamlit (if needed)

**Admin page not accessible?**
- Login with username `admin`
- Check authentication is working
- Verify admin check in page code

**Changes not saving?**
- Check file permissions
- Verify JSON is valid
- Look for error messages

## Example: Sample Company Use Case

**Zepto-style feature toggling:**
```json
{
  "features": {
    "grocery": {"enabled": true},
    "cafe": {"enabled": false},  // Temporarily disabled
    "fresh": {"enabled": true}
  }
}
```

Users won't see "Cafe" option until you change `"enabled": true`.

---

**Questions?** Contact your system administrator.
