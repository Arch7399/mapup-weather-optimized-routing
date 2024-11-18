# Import necessary libraries for mathematical operations, data handling, and visualization
from dataclasses import dataclass  # For creating data classes with less boilerplate
from typing import List, Set  # For type hints
import folium  # For creating interactive maps
from shapely.geometry import Point, LineString  # For geometric operations
import geopandas as gpd  # For handling geospatial data
import math  # For mathematical calculations
import random  # For generating random values


@dataclass
class Coordinate:
    """
    Represents a geographical coordinate with latitude and longitude.
    Uses @dataclass decorator to automatically generate __init__, __repr__, etc.
    """

    lat: float  # Latitude in degrees
    lon: float  # Longitude in degrees

    def distance_to(self, other: "Coordinate") -> float:
        """
        Calculate the great-circle distance between two coordinates using the Haversine formula.

        Args:
            other (Coordinate): The target coordinate to calculate distance to

        Returns:
            float: Distance in kilometers between the two points
        """
        R = 6371  # Earth's radius in kilometers
        # Convert latitude and longitude to radians for trigonometric calculations
        lat1, lon1 = math.radians(self.lat), math.radians(self.lon)
        lat2, lon2 = math.radians(other.lat), math.radians(other.lon)

        # Calculate differences in coordinates
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Haversine formula implementation
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def to_point(self):
        """
        Convert coordinate to Shapely Point object for geometric operations.
        Note: Shapely uses (longitude, latitude) order, opposite of typical (latitude, longitude)

        Returns:
            Point: Shapely Point object representing the coordinate
        """
        return Point(self.lon, self.lat)


class Vehicle:
    """
    Represents a vehicle moving along a route with position tracking and movement capabilities.
    """

    def __init__(self, position: Coordinate):
        """
        Initialize vehicle with starting position and movement parameters.

        Args:
            position (Coordinate): Starting position of the vehicle
        """
        self.position = position  # Current position
        self.total_distance = 0  # Total distance traveled in kilometers
        self.speed = 80  # Average speed in km/h
        self.current_waypoint_index = 0  # Index of last passed waypoint
        self.next_waypoint_index = 1  # Index of next waypoint
        self.reached_destination = False  # Flag indicating if destination is reached

    def update_position(self, route: "EnhancedRoute", hours_passed: int) -> None:
        """
        Update vehicle position based on time passed and route information.
        Calculates new position by interpolating between waypoints based on speed and time.

        Args:
            route (EnhancedRoute): Current route being followed
            hours_passed (int): Number of hours passed in simulation
        """
        # If already at destination, no need to update
        if self.reached_destination:
            return

        # Calculate how far the vehicle should have traveled
        distance_per_hour = self.speed
        target_distance = hours_passed * distance_per_hour

        waypoints = route.waypoints

        # Continue moving through waypoints until target distance is reached
        while self.next_waypoint_index < len(waypoints):
            segment_start = waypoints[self.current_waypoint_index]
            segment_end = waypoints[self.next_waypoint_index]

            # Calculate length of current segment
            segment_distance = segment_start.distance_to(segment_end)

            # Check if target distance falls within current segment
            if self.total_distance + segment_distance > target_distance:
                # Calculate position within current segment through linear interpolation
                remaining_distance = target_distance - self.total_distance
                progress = remaining_distance / segment_distance

                # Interpolate new position
                new_lat = (
                    segment_start.lat + (segment_end.lat - segment_start.lat) * progress
                )
                new_lon = (
                    segment_start.lon + (segment_end.lon - segment_start.lon) * progress
                )

                self.position = Coordinate(new_lat, new_lon)
                self.total_distance = target_distance
                return

            # Move to next segment if target distance not reached
            self.total_distance += segment_distance
            self.current_waypoint_index += 1
            self.next_waypoint_index += 1

            # Check if reached final waypoint
            if self.next_waypoint_index >= len(waypoints):
                self.position = waypoints[-1]
                self.reached_destination = True
                return


class WeatherPattern:
    """
    Represents a dynamic weather system that moves and affects route conditions.
    Includes random variations in movement and intensity for realistic simulation.
    """

    def __init__(
        self,
        center: Coordinate,
        severity: float,
        radius: float,
        movement_speed: float,
        movement_direction: float,
    ):
        """
        Initialize weather pattern with its characteristics.

        Args:
            center (Coordinate): Center point of weather system
            severity (float): Intensity of weather system (0-1)
            radius (float): Radius of effect in kilometers
            movement_speed (float): Speed of movement in km/h
            movement_direction (float): Direction of movement in degrees
        """
        self.center = center
        self.severity = severity
        self.radius = radius
        self.movement_speed = movement_speed
        self.movement_direction = movement_direction
        self.initial_center = center  # Store initial position
        self.time_since_last_change = 0  # Track time for pattern changes
        self.change_interval = random.randint(2, 5)  # Random interval for changes

    def update_position(self, hours: float):
        """
        Update weather pattern position and characteristics with random variations.
        Simulates realistic weather pattern movement and evolution.

        Args:
            hours (float): Time passed in simulation
        """
        self.time_since_last_change += 1

        # Periodically update weather pattern characteristics
        if self.time_since_last_change >= self.change_interval:
            # Randomly change direction within 90 degrees
            direction_change = random.uniform(-90, 90)
            self.movement_direction = (self.movement_direction + direction_change) % 360

            # Adjust speed randomly
            self.movement_speed *= random.uniform(0.7, 1.4)

            # Modify severity and radius with constraints
            self.severity = min(1.0, max(0.3, self.severity * random.uniform(0.8, 1.3)))
            self.radius *= random.uniform(0.9, 1.2)

            # Reset change timer
            self.time_since_last_change = 0
            self.change_interval = random.randint(2, 5)

        # Add random variations to movement
        turbulence_lat = random.uniform(-0.2, 0.2)
        turbulence_lon = random.uniform(-0.2, 0.2)

        # Calculate base movement vector
        distance = self.movement_speed * hours
        direction_rad = math.radians(self.movement_direction)
        dy = distance * math.cos(direction_rad)
        dx = distance * math.sin(direction_rad)

        # Convert movement to coordinate changes
        # Note: 111 km is approximate distance of 1 degree of latitude
        lat_change = (dy / 111) + turbulence_lat
        lon_change = (
            dx / (111 * math.cos(math.radians(self.center.lat)))
        ) + turbulence_lon

        # Update center position
        self.center = Coordinate(
            self.initial_center.lat + lat_change, self.initial_center.lon + lon_change
        )

    def calculate_impact(
        self, point: Coordinate, vehicle_position: Coordinate, destination: Coordinate
    ) -> float:
        """
        Calculate weather impact on a specific point considering vehicle position and destination.

        Args:
            point (Coordinate): Point to calculate impact for
            vehicle_position (Coordinate): Current vehicle position
            destination (Coordinate): Final destination

        Returns:
            float: Impact value between 0 and 1
        """
        # Check if point is in relevant area
        if not self._is_in_active_area(point, vehicle_position, destination):
            return 0.0

        # Calculate distance from weather center
        point_dist = point.distance_to(self.center)
        if point_dist > self.radius:
            return 0.0

        # Calculate base impact using inverse square law
        base_impact = self.severity * (1 - (point_dist / self.radius) ** 2)

        # Add oscillating factors for more dynamic weather effects
        time_factor = (
            1
            + 0.3 * math.sin(point_dist * 0.1)
            + 0.2 * math.cos(point_dist * 0.05)
            + 0.15 * math.sin(self.center.lat * 0.1)
        )

        return min(1.0, base_impact * time_factor)

    def _is_in_active_area(
        self, point: Coordinate, vehicle_position: Coordinate, destination: Coordinate
    ) -> bool:
        """
        Determine if a point is in the relevant area between vehicle and destination.
        Uses triangle inequality with buffer to define relevant area.

        Args:
            point (Coordinate): Point to check
            vehicle_position (Coordinate): Current vehicle position
            destination (Coordinate): Final destination

        Returns:
            bool: True if point is in active area
        """
        total_distance = vehicle_position.distance_to(destination)
        dist_to_vehicle = point.distance_to(vehicle_position)
        dist_to_dest = point.distance_to(destination)

        # Allow 20% buffer zone around direct path
        buffer = total_distance * 0.2
        return dist_to_vehicle + dist_to_dest <= total_distance + buffer


# Note: The rest of the code continues with similar detailed commenting...
# I'll continue with the remaining classes and functions if you'd like,
# but this gives you a good idea of the commenting style and level of detail.
# Would you like me to continue with the rest of the code?
class EnhancedRoute:
    """
    Represents a complete route with multiple waypoints and associated characteristics.
    Includes risk assessment and cost calculation capabilities.
    """

    def __init__(
        self,
        id: int,
        name: str,
        waypoints: List[Coordinate],
        base_travel_time: float,
        base_cost: float,
        flood_susceptibility: float,
        traffic_factor: float = 1.0,
        adjacent_routes: Set[int] = set(),  # IDs of connected routes
    ):
        """
        Initialize route with all necessary parameters and characteristics.

        Args:
            id (int): Unique identifier for the route
            name (str): Descriptive name of the route
            waypoints (List[Coordinate]): Ordered list of route points
            base_travel_time (float): Expected travel time in hours without delays
            base_cost (float): Base cost in dollars without adjustments
            flood_susceptibility (float): Route's vulnerability to flooding (0-1)
            traffic_factor (float): Traffic density multiplier (default: 1.0)
            adjacent_routes (Set[int]): IDs of routes that can be switched to
        """
        self.id = id
        self.name = name
        self.waypoints = waypoints
        self.base_travel_time = base_travel_time
        self.base_cost = base_cost
        self.flood_susceptibility = flood_susceptibility
        self.traffic_factor = traffic_factor
        self.current_risk = 0.0  # Current calculated risk level
        self.geometry = self._create_geometry()  # LineString for mapping
        self.adjacent_routes = adjacent_routes

    def _create_geometry(self):
        """
        Create a LineString geometry from waypoints for mapping purposes.
        Converts coordinates to format suitable for GeoJSON representation.

        Returns:
            LineString: Shapely LineString object representing the route
        """
        # Convert coordinates to (longitude, latitude) pairs for GeoJSON
        points = [(wp.lon, wp.lat) for wp in self.waypoints]
        return LineString(points)

    def calculate_effective_cost(self, current_position: Coordinate) -> float:
        """
        Calculate the total effective cost of completing the route from current position.
        Considers distance, time, risk factors, and traffic conditions.

        Args:
            current_position (Coordinate): Current vehicle position

        Returns:
            float: Total effective cost in dollars
        """
        # Find nearest waypoint to current position
        remaining_index = self._find_nearest_waypoint(current_position)
        remaining_waypoints = self.waypoints[remaining_index:]

        # Calculate remaining distance along route
        remaining_distance = sum(
            wp1.distance_to(wp2)
            for wp1, wp2 in zip(remaining_waypoints[:-1], remaining_waypoints[1:])
        )

        # Calculate base costs
        distance_cost = remaining_distance * 2  # $2 per kilometer
        time_cost = self.base_travel_time * 50  # $50 per hour

        # Apply risk and traffic multipliers
        risk_multiplier = 1 + (self.current_risk * 2)  # Higher risk doubles cost
        traffic_multiplier = self.traffic_factor

        # Compute final adjusted cost
        effective_cost = (
            (distance_cost + time_cost) * risk_multiplier * traffic_multiplier
        )

        return effective_cost

    def _find_nearest_waypoint(self, position: Coordinate) -> int:
        """
        Find the index of the nearest waypoint to given position.

        Args:
            position (Coordinate): Position to find nearest waypoint to

        Returns:
            int: Index of nearest waypoint in waypoints list
        """
        distances = [position.distance_to(wp) for wp in self.waypoints]
        return distances.index(min(distances))

    def to_gdf(self):
        """
        Convert route to GeoDataFrame for geographic visualization.

        Returns:
            GeoDataFrame: Route data in geographic format
        """
        return gpd.GeoDataFrame(
            {"id": [self.id], "name": [self.name], "risk": [self.current_risk]},
            geometry=[self.geometry],
            crs="EPSG:4326",  # WGS84 coordinate system
        )


def run_enhanced_simulation(
    hours: int, routes: List[EnhancedRoute], start: Coordinate, end: Coordinate
):
    """
    Main simulation function that runs the entire route optimization process.

    Args:
        hours (int): Maximum simulation duration in hours
        routes (List[EnhancedRoute]): Available routes
        start (Coordinate): Starting position
        end (Coordinate): Destination position

    Returns:
        List[dict]: Hourly simulation results with position and cost data
    """
    results = []
    vehicle = Vehicle(start)
    # Start with the most cost-effective route
    current_route = min(routes, key=lambda r: r.calculate_effective_cost(start))
    weather_patterns = create_sample_weather_patterns()

    # Main simulation loop
    for hour in range(hours):
        # Check if destination reached
        if vehicle.reached_destination:
            print("Reached destination!")
            break

        print(f"Processing hour {hour+1}/{hours}...")

        # Update all weather patterns
        for pattern in weather_patterns:
            pattern.update_position(hour)

        # Recalculate risks for all routes
        for route in routes:
            max_risk = 0.0
            for point in route.waypoints:
                # Base risk from flood susceptibility
                point_risk = route.flood_susceptibility
                # Add weather impacts
                for weather in weather_patterns:
                    point_risk += weather.calculate_impact(point, vehicle.position, end)
                max_risk = max(max_risk, point_risk)
            route.current_risk = max_risk

        # Check for better adjacent routes
        best_adjacent_route = find_best_adjacent_route(
            current_route, routes, vehicle.position
        )

        # Switch routes if better option found
        if best_adjacent_route != current_route:
            print(f"Switching from {current_route.name} to {best_adjacent_route.name}")
            current_progress = vehicle.total_distance
            current_route = best_adjacent_route
            # Maintain progress when switching routes
            vehicle = Vehicle(vehicle.position)
            vehicle.total_distance = current_progress

        # Move vehicle forward
        vehicle.update_position(current_route, hour + 1)

        # Generate visualization
        m = create_visualization(
            routes,
            weather_patterns,
            vehicle.position,
            end,
            current_route,
            current_route,
        )

        # Save hourly map
        filename = f"enhanced_route_map_hour_{hour:02d}.html"
        m.save(filename)
        print(f"Saved map: {filename}")

        # Store results
        results.append(
            {
                "hour": hour,
                "route": current_route,
                "position": vehicle.position,
                "map": m,
                "cost": current_route.calculate_effective_cost(vehicle.position),
            }
        )

        # Log progress
        print(
            f"Hour {hour+1}: Position: ({vehicle.position.lat:.4f}, {vehicle.position.lon:.4f}) "
            f"on {current_route.name}"
        )

    return results


def create_visualization(
    routes: List[EnhancedRoute],
    weather_patterns: List[WeatherPattern],
    current_position: Coordinate,
    destination: Coordinate,
    optimal_route: EnhancedRoute,
    current_route: EnhancedRoute,
) -> folium.Map:
    """
    Create an interactive map visualization of the current simulation state.

    Args:
        routes: All available routes
        weather_patterns: Active weather systems
        current_position: Vehicle's current location
        destination: Target location
        optimal_route: Best route based on current conditions
        current_route: Route currently being followed

    Returns:
        folium.Map: Interactive map with all visualization elements
    """
    # Initialize base map centered on vehicle position
    m = folium.Map(
        location=[current_position.lat, current_position.lon],
        zoom_start=6,
        tiles="cartodbpositron",  # Clean, minimal map style
    )

    # Draw all routes with appropriate styling
    for route in routes:
        # Determine route color based on status and risk
        if route == current_route and route == optimal_route:
            color = "green"  # Optimal current route
        elif route == current_route:
            color = "yellow"  # Non-optimal current route
        else:
            color = "red" if route.current_risk > 0.5 else "blue"  # Other routes

        # Set line style parameters
        opacity = 1.0 if route == current_route else 0.5
        weight = 4 if route == current_route else 2

        # Draw route line
        coords = [(wp.lat, wp.lon) for wp in route.waypoints]
        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=opacity,
            popup=f"Route {route.id}: {route.name}<br>Risk: {route.current_risk:.2f}",
        ).add_to(m)

    # Visualize weather patterns
    for weather in weather_patterns:
        folium.Circle(
            location=[weather.center.lat, weather.center.lon],
            radius=weather.radius * 1000,  # Convert km to meters
            color="blue",
            fill=True,
            opacity=weather.severity * 0.5,
            popup=f"Weather System<br>Severity: {weather.severity:.2f}",
        ).add_to(m)

    # Add vehicle marker
    folium.Marker(
        [current_position.lat, current_position.lon],
        popup=f"Current Position<br>Route: {current_route.name}",
        icon=folium.Icon(color="green", icon="truck", prefix="fa"),
    ).add_to(m)

    # Add start marker
    folium.Marker(
        [routes[0].waypoints[0].lat, routes[0].waypoints[0].lon],
        popup="Denver (Start)",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    # Add destination marker
    folium.Marker(
        [destination.lat, destination.lon],
        popup="Minneapolis (Destination)",
        icon=folium.Icon(color="red", icon="flag"),
    ).add_to(m)

    # Add interactive legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: 160px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;
                border-radius: 5px;">
                <p><strong>Route Colors:</strong></p>
                <p><i class="fa fa-line" style="color:green;"></i> Current Optimal Route</p>
                <p><i class="fa fa-line" style="color:yellow;"></i> Current Non-optimal Route</p>
                <p><i class="fa fa-line" style="color:blue;"></i> Alternative Route (Low Risk)</p>
                <p><i class="fa fa-line" style="color:red;"></i> Alternative Route (High Risk)</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# Helper Functions for Route Creation and Simulation


def create_sample_routes() -> List[EnhancedRoute]:
    """Create sample routes between Denver and Minneapolis with adjacency information"""
    denver = Coordinate(39.7392, -104.9903)
    minneapolis = Coordinate(44.9778, -93.2650)

    # Route 1: I-76 to I-80 (Northern Route)
    route1_waypoints = [
        denver,
        Coordinate(40.8258, -102.7677),  # Sterling, CO
        Coordinate(41.1403, -100.7601),  # North Platte, NE
        Coordinate(41.1399, -98.0098),  # Grand Island, NE
        Coordinate(41.2565, -95.9345),  # Omaha, NE
        Coordinate(41.5868, -93.6250),  # Des Moines, IA
        Coordinate(43.6666, -93.3499),  # Albert Lea, MN
        minneapolis,
    ]

    # Route 2: I-25 to I-90 (Wyoming Route)
    route2_waypoints = [
        denver,
        Coordinate(41.1456, -104.8019),  # Cheyenne, WY
        Coordinate(43.0760, -106.3319),  # Casper, WY
        Coordinate(44.0805, -103.2310),  # Rapid City, SD
        Coordinate(43.8791, -99.3267),  # Chamberlain, SD
        Coordinate(43.5460, -96.7313),  # Sioux Falls, SD
        Coordinate(43.8791, -93.9619),  # Worthington, MN
        minneapolis,
    ]

    # Route 3: I-70 to I-35 (Central Route)
    route3_waypoints = [
        denver,
        Coordinate(39.2639, -103.6925),  # Limon, CO
        Coordinate(39.1889, -100.8534),  # Oakley, KS
        Coordinate(38.8403, -97.6114),  # Salina, KS
        Coordinate(39.0997, -94.5786),  # Kansas City, MO
        Coordinate(41.5868, -93.6250),  # Des Moines, IA
        Coordinate(43.6666, -93.3499),  # Albert Lea, MN
        minneapolis,
    ]

    # Route 4: Nebraska Secondary Route
    route4_waypoints = [
        denver,
        Coordinate(40.8258, -102.7677),  # Sterling, CO
        Coordinate(40.1989, -100.6267),  # McCook, NE
        Coordinate(41.1399, -98.0098),  # Grand Island, NE
        Coordinate(41.8780, -93.0977),  # Marshalltown, IA
        Coordinate(43.6666, -93.3499),  # Albert Lea, MN
        minneapolis,
    ]

    # Route 5: Southern Kansas Route
    route5_waypoints = [
        denver,
        Coordinate(39.3061, -102.2693),  # Burlington, CO
        Coordinate(39.3486, -101.7107),  # Goodland, KS
        Coordinate(39.3500, -99.3267),  # Hays, KS
        Coordinate(39.0473, -95.6752),  # Topeka, KS
        Coordinate(41.5868, -93.6250),  # Des Moines, IA
        Coordinate(43.6666, -93.3499),  # Albert Lea, MN
        minneapolis,
    ]

    routes = [
        EnhancedRoute(
            id=1,
            name="Northern I-80 Route",
            waypoints=route1_waypoints,
            base_travel_time=15.5,
            base_cost=950,
            flood_susceptibility=0.4,
            traffic_factor=1.1,
            adjacent_routes={2, 4},
        ),
        EnhancedRoute(
            id=2,
            name="Northern Wyoming Route",
            waypoints=route2_waypoints,
            base_travel_time=16.5,
            base_cost=1050,
            flood_susceptibility=0.2,
            traffic_factor=0.9,
            adjacent_routes={1, 3},
        ),
        EnhancedRoute(
            id=3,
            name="Central I-70 Route",
            waypoints=route3_waypoints,
            base_travel_time=15.0,
            base_cost=1000,
            flood_susceptibility=0.5,
            traffic_factor=1.2,
            adjacent_routes={2, 4, 5},
        ),
        EnhancedRoute(
            id=4,
            name="Nebraska Secondary Route",
            waypoints=route4_waypoints,
            base_travel_time=15.8,
            base_cost=925,
            flood_susceptibility=0.6,
            traffic_factor=0.95,
            adjacent_routes={1, 3, 5},
        ),
        EnhancedRoute(
            id=5,
            name="Southern Kansas Route",
            waypoints=route5_waypoints,
            base_travel_time=16.2,
            base_cost=975,
            flood_susceptibility=0.3,
            traffic_factor=1.0,
            adjacent_routes={3, 4},
        ),
    ]

    return routes


def create_sample_weather_patterns() -> List[WeatherPattern]:
    """
    Generate random weather patterns for simulation.
    Creates 6-8 weather systems with varying characteristics.

    Returns:
        List[WeatherPattern]: List of weather patterns
    """
    patterns = []
    num_patterns = random.randint(6, 8)

    for _ in range(num_patterns):
        # Generate random parameters for each weather pattern
        lat = random.uniform(39.0, 45.0)  # Denver-Minneapolis latitude range
        lon = random.uniform(-105.0, -93.0)  # Denver-Minneapolis longitude range

        patterns.append(
            WeatherPattern(
                center=Coordinate(lat, lon),
                severity=random.uniform(0.6, 1.0),
                radius=random.uniform(150, 300),
                movement_speed=random.uniform(25, 45),
                movement_direction=random.uniform(0, 360),
            )
        )

    return patterns


def find_best_adjacent_route(
    current_route: EnhancedRoute,
    all_routes: List[EnhancedRoute],
    current_position: Coordinate,
) -> EnhancedRoute:
    """
    Find the most cost-effective route among current and adjacent routes.

    Args:
        current_route: Currently active route
        all_routes: All available routes
        current_position: Vehicle's current position

    Returns:
        EnhancedRoute: Best route based on current conditions
    """
    # Get all routes that can be switched to
    adjacent_routes = [r for r in all_routes if r.id in current_route.adjacent_routes]
    adjacent_routes.append(current_route)

    # Return route with lowest effective cost
    return min(
        adjacent_routes, key=lambda r: r.calculate_effective_cost(current_position)
    )


def run_simulation_demo():
    """
    Run a complete demonstration of the route optimization system.
    Sets up routes, runs simulation, and handles output.
    """
    print("Starting simulation with 5 routes between Denver and Minneapolis...")

    try:
        # Create routes
        routes = create_sample_routes()

        # Define endpoints
        start = Coordinate(39.7392, -104.9903)  # Denver
        end = Coordinate(44.9778, -93.2650)  # Minneapolis

        # Run simulation
        results = run_enhanced_simulation(20, routes, start, end)
        print("\nSimulation completed successfully!")
        print(f"Generated {len(results)} hourly maps")

    except Exception as e:
        print(f"Error during simulation: {e}")
        raise  # Re-raise the exception to see the full traceback


if __name__ == "__main__":
    run_simulation_demo()
