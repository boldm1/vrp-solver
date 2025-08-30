import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import requests

# OSRM API endpoints for the public demo server.
# For heavy use, you should host your own OSRM instance.
# See: https://github.com/Project-OSRM/osrm-backend
OSRM_TABLE_URL = "http://router.project-osrm.org/table/v1/driving/"

# Nominatim API endpoint for geocoding.
# See: https://operations.osmfoundation.org/policies/nominatim/
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"


def get_coords_for_addresses(
    addresses: List[str],
) -> Dict[str, Tuple[float, float]]:
    """
    Geocodes a list of addresses to (longitude, latitude) coordinates using Nominatim.
    Returns a dictionary mapping successfully geocoded addresses to their coordinates.
    """
    address_coords = {}
    print("Geocoding addresses using Nominatim...")
    # Nominatim requires a custom User-Agent and has a rate limit of 1 req/sec.
    headers = {"User-Agent": "VrpSolver/1.0 (github.com/mattbold/vrp-solver)"}

    for address in addresses:
        try:
            params = {"q": address, "format": "jsonv2"}
            response = requests.get(
                NOMINATIM_SEARCH_URL, params=params, headers=headers
            )
            response.raise_for_status()
            data = response.json()

            if data:
                # Take the first result
                location = data[0]
                # OSRM expects (longitude, latitude)
                lon, lat = float(location["lon"]), float(location["lat"])
                address_coords[address] = (lon, lat)
                print(f"  ✓ {address} -> ({lon:.6f}, {lat:.6f})")
            else:
                print(f"  ✗ Could not find coordinates for: {address}. Skipping.")

            # Respect Nominatim's rate limit of 1 request per second
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"Error geocoding '{address}': {e}. Skipping.")
        except (IndexError, KeyError):
            print(f"  ✗ No valid results found for: {address}. Skipping.")

    return address_coords


def get_distance_matrix(coords: List[Tuple[float, float]]) -> List[List[float]]:
    """
    Fetches a distance matrix (in meters) for a list of coordinates using OSRM.
    """
    print("\nFetching distance matrix from OSRM...")
    # Format coordinates for the OSRM API URL: {lon},{lat};{lon},{lat};...
    coords_str = ";".join([f"{lon},{lat}" for lon, lat in coords])
    url = f"{OSRM_TABLE_URL}{coords_str}?annotations=distance"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data["code"] == "Ok":
            print("  ✓ Successfully retrieved distance matrix.")
            # OSRM returns distances in meters. We'll round to integers.
            return [[round(d) for d in row] for row in data["distances"]]
        else:
            print(f"  ✗ OSRM API error: {data.get('message', 'Unknown error')}")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching distance matrix: {e}")
        sys.exit(1)


def create_instance_json(
    depot_address: str,
    customer_addresses: List[str],
    vehicle_definitions: List[Dict],
    customer_demands: List[int],
) -> Dict | None:
    """
    Creates a VRP instance JSON object from real-world addresses.
    Returns None if the depot cannot be geocoded.
    """
    initial_all_addresses = [depot_address] + customer_addresses
    address_to_demand_map = dict(zip(customer_addresses, customer_demands))

    address_coords = get_coords_for_addresses(initial_all_addresses)

    if depot_address not in address_coords:
        print(
            f"\nCritical error: Depot '{depot_address}' could not be geocoded. Aborting."
        )
        return None

    # The depot was found. Now filter customers to only include those that
    # were successfully geocoded.
    final_customer_addresses = [
        addr for addr in customer_addresses if addr in address_coords
    ]
    final_customer_demands = [
        address_to_demand_map[addr] for addr in final_customer_addresses
    ]

    # The final list of locations, with the depot at index 0.
    final_coords = [address_coords[depot_address]] + [
        address_coords[addr] for addr in final_customer_addresses
    ]

    if len(final_customer_addresses) < len(customer_addresses):
        print(
            f"\nProceeding with {len(final_customer_addresses)} of "
            f"{len(customer_addresses)} customers that were successfully geocoded."
        )

    dist_matrix = get_distance_matrix(final_coords)

    # Create the 'locations' list for the JSON file
    locations_json = []
    locations_json.append({"name": depot_address, "coords": final_coords[0]})
    for i, address in enumerate(final_customer_addresses):
        locations_json.append(
            {
                "name": address,
                "coords": final_coords[i + 1],
                "demand": final_customer_demands[i],
            }
        )

    # Create the 'depots' list for the JSON file
    depots_json = [{"name": depot_address, "fleet": vehicle_definitions}]

    # Assemble the final JSON structure
    instance_data = {
        "locations": locations_json,
        "depots": depots_json,
        "distance_matrix": dist_matrix,
    }
    return instance_data


def main():
    parser = argparse.ArgumentParser(
        description="Create a VRP instance JSON file from a list of addresses using OSRM."
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save the generated VRP instance JSON file (e.g., 'data/london_instance.json').",
    )
    args = parser.parse_args()

    # --- Define your real-world problem here ---
    depot = "Cabot Tower, Bristol"
    customers = [
        "12 Park Street, Bristol",
        "45 Gloucester Road, Bishopston, Bristol",
        "8 Clifton Down, Clifton, Bristol",
        "22 North Street, Bedminster, Bristol",
        "101 Whiteladies Road, Clifton, Bristol",
        "3 The Paragon, Clifton, Bristol",
        "76 Stokes Croft, Bristol",
        "19 King Street, Bristol",
        "50 Corn Street, Bristol",
        "34 Queen Square, Bristol",
        "88 Temple Meads, Bristol",
        "15 Redcliffe Hill, Redcliffe, Bristol",
        "29 East Street, Bedminster, Bristol",
        "67 Chandos Road, Redland, Bristol",
        "4 Cotham Hill, Cotham, Bristol",
        "112 St. Michael's Hill, Bristol",
        "23 Welsh Back, Bristol",
        "5 Harbourside, Bristol",
        "14 Hotwell Road, Hotwells, Bristol",
        "31 Coronation Road, Southville, Bristol",
        "9 Berkeley Square, Clifton, Bristol",
        "44 Park Row, Bristol",
        "18 Triangle West, Clifton, Bristol",
        "7 St. Stephen's Street, Bristol",
        "123 Fishponds Road, Fishponds, Bristol",
        "38 Stapleton Road, Easton, Bristol",
        "55 Church Road, Redfield, Bristol",
        "2 Sandy Park Road, Brislington, Bristol",
        "16 High Street, Keynsham, Bristol",
        "42 The Downs, Sneyd Park, Bristol",
        "8 Portishead High Street, Portishead",
        "17 Clevedon Seafront, Clevedon",
        "22 Thornbury High Street, Thornbury",
        "9 Yate Shopping Centre, Yate",
        "1 Henleaze Road, Henleaze, Bristol",
        "5 Westbury Hill, Westbury-on-Trym, Bristol",
        "30 Canford Lane, Westbury-on-Trym, Bristol",
        "14 Shirehampton High Street, Shirehampton, Bristol",
        "25 Avonmouth Road, Avonmouth, Bristol",
        "6 Filton Avenue, Filton, Bristol",
        "11 Long Ashton Road, Long Ashton, Bristol",
        "47 Bath Road, Totterdown, Bristol",
        "8 Wells Road, Knowle, Bristol",
        "19 Zetland Road, Redland, Bristol",
        "33 Picton Street, Montpelier, Bristol",
        "7 Royal York Crescent, Clifton, Bristol",
        "21 Baldwin Street, Bristol",
        "10 Union Street, Bristol",
        "5 Cabot Circus, Bristol",
        "16 College Green, Bristol",
    ]
    # Define a fleet of four identical vehicles
    fleet = [
        {"capacity": 50, "range_kms": 0, "fixed_cost": 50},
        {"capacity": 50, "range_kms": 0, "fixed_cost": 50},
        {"capacity": 50, "range_kms": 0, "fixed_cost": 50},
        {"capacity": 50, "range_kms": 0, "fixed_cost": 50},
    ]
    # Demands for each customer
    demands = [
        2,
        5,
        3,
        6,
        4,
        2,
        5,
        3,
        1,
        4,
        6,
        2,
        5,
        3,
        4,
        2,
        5,
        3,
        6,
        4,
        2,
        5,
        3,
        1,
        4,
        6,
        2,
        5,
        3,
        4,
        2,
        5,
        3,
        6,
        4,
        2,
        5,
        3,
        1,
        4,
        6,
        2,
        5,
        3,
        4,
        2,
        5,
        3,
        6,
        4,
    ]

    if len(customers) != len(demands):
        print("Error: The number of customers must match the number of demands.")
        sys.exit(1)

    # --- Generate and save the instance ---
    print(f"Generating VRP instance for {len(customers)} customers and 1 depot.")
    instance_json = create_instance_json(depot, customers, fleet, demands)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving instance to {args.output_path}...")
    with open(args.output_path, "w") as f:
        json.dump(instance_json, f, indent=4)

    print("Done!")


if __name__ == "__main__":
    main()
