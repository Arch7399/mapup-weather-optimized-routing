# Dynamic Route Optimization System with Weather Impact Analysis

## Overview
This project implements an advanced route optimization system for fleet management, inspired by Mapup's fleet routing optimization solutions. While the original focus was on reducing toll expenses and fuel costs, this implementation adds a randomized weather simulation layer to provide more realistic and dynamic route optimization.

## Key Features
- **Multi-Route Analysis**: Simulates and analyzes 5 distinct routes between Denver and Minneapolis
- **Dynamic Weather Patterns**: Incorporates 6-8 moving weather systems that affect route safety and efficiency
- **Real-Time Route Adjustment**: Enables dynamic route switching based on changing conditions
- **Interactive Visualization**: Generates hourly maps showing route conditions and weather patterns
- **Cost Optimization**: Considers multiple factors including base cost, weather risk, and traffic conditions

## Technical Components

### Core Classes
1. **Coordinate**
   - Handles geographical positions (latitude/longitude)
   - Implements distance calculations using Haversine formula
   - Converts coordinates to geometric points for visualization

2. **WeatherPattern**
   - Simulates dynamic weather systems
   - Features:
     - Variable severity levels
     - Configurable radius of impact
     - Dynamic movement patterns
     - Random direction changes
     - Turbulence simulation

3. **EnhancedRoute**
   - Defines route characteristics
   - Manages:
     - Base travel time
     - Cost calculations
     - Flood susceptibility
     - Traffic factors
     - Adjacent route connections

## Route Network
The system models five major routes between Denver and Minneapolis:
1. Northern I-80 Route
2. Northern Wyoming Route
3. Central I-70 Route
4. Nebraska Secondary Route
5. Southern Kansas Route

Each route includes:
- Detailed waypoints
- Base costs and travel times
- Risk factors
- Traffic conditions
- Connections to adjacent routes

## Simulation Process

### Initialization
1. Creates route network with predefined characteristics
2. Generates dynamic weather patterns
3. Initializes vehicle at starting position (Denver)

### Hourly Updates
1. **Weather System Updates**
   - Updates positions of all weather patterns
   - Adjusts severity and characteristics
   - Applies turbulence and random variations

2. **Risk Assessment**
   - Calculates current risk levels for all routes
   - Considers:
     - Weather impact
     - Flood susceptibility
     - Traffic conditions

3. **Route Optimization**
   - Evaluates current route efficiency
   - Checks adjacent routes for better alternatives
   - Makes routing decisions based on multiple factors

4. **Visualization**
   - Generates interactive maps showing:
     - Active weather patterns
     - Route risk levels
     - Optimal path selection

## Implementation Details

### Cost Calculation
The system calculates effective costs considering:
- Distance-based costs ($2 per km)
- Time-based costs ($50 per hour)
- Risk multipliers based on weather conditions
- Traffic impact factors

### Weather Impact
Weather patterns affect routes through:
- Severity levels (0.3 to 1.0)
- Variable radius (150-300 km)
- Dynamic movement (25-45 km/h)
- Direction changes every 2-5 hours

## Visualization Features
- Interactive Folium maps
- Color-coded routes based on risk levels
- Weather pattern visualization
- Clear route distinction between:
  - Current optimal route (green)
  - Current non-optimal route (yellow)
  - Alternative routes (blue/red based on risk)
  
## Clone Project

```bash
git clone https://github.com/Arch7399/mapup-weather-optimized-routing.git
```
## Usage
# Run the complete simulation
```python 
dynamic-weather-route-mapping.py
```

## Dependencies
- folium
- shapely
- geopandas
- math
- random
```python
pip install folium shapely geopandas math random
```

## Origin and Enhancement
This project builds upon Mapup's fleet routing optimization concept by adding:
1. Dynamic weather pattern simulation
2. Real-time route risk assessment
3. Enhanced visualization features

The weather simulation addition provides a more comprehensive approach to route optimization, considering not just static factors like tolls and fuel costs, but also dynamic environmental conditions that can significantly impact route efficiency and safety.

## Output
The system generates hourly HTML maps showing the simulation progress(sample inside the assets folder here), saved as:
- `enhanced_route_map_hour_XX.html`
- Where XX represents the hour number (00-19) : (travel hours can be changed inside the script)

Each map provides a complete visualization of the current situation, including weather patterns and route conditions.

