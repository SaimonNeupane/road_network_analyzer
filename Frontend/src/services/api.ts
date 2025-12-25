import axios from "axios";
import type { LocationData } from "../types/location"; // Ensure this type is imported

const base_url = axios.create({
  baseURL: "http://localhost:8000",
});

export const fetchLocations = async (dis: string, cat: string) => {
  // Sending category as a query param (e.g., /jhapa?category=park)
  // If 'all', we might not want to send the param, or send 'all' depending on backend logic
  const params: Record<string, string> = {};
  if (cat && cat !== "all") {
    params.category = cat;
  }

  // Assuming your backend route is /{district_name}
  return await base_url.get<LocationData[]>(`/${dis}`, {
    params: params
  });
};
