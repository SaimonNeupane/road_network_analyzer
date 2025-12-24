import type { LocationData } from "../types/location";

export const fetchLocations = async (
  query: string,
  category: string
): Promise<LocationData[]> => {
  const res = await fetch(
    `/api/locations?search=${query}&category=${category}`
  );
  return res.json();
};
