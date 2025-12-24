import type { LocationData } from "../types/location";

export const fetchLocations = async (
  search: string,
  category: string
): Promise<LocationData[]> => {
  // Mock data for demonstration - replace with actual API call
  // Example: const response = await fetch(`/api/locations?search=${search}&category=${category}`);
  // return response.json();

  await new Promise((resolve) => setTimeout(resolve, 500)); // Simulate API delay

  const mockLocations: LocationData[] = [
    {
      id: "1",
      name: "Kathmandu University",
      latitude: 27.6194,
      longitude: 85.5388,
      description: "A university in Dhulikhel, Nepal",
      category: "Education",
    },
    {
      id: "2",
      name: "Patan Durbar Square",
      latitude: 27.6734,
      longitude: 85.3256,
      description: "Historic palace square",
      category: "Tourism",
    },
    {
      id: "3",
      name: "Boudhanath Stupa",
      latitude: 27.7215,
      longitude: 85.362,
      description: "UNESCO World Heritage Site",
      category: "Religious",
    },
  ];

  // Filter based on search and category
  return mockLocations.filter((loc) => {
    const matchesSearch =
      search === "" ||
      loc.name.toLowerCase().includes(search.toLowerCase()) ||
      loc.description.toLowerCase().includes(search.toLowerCase());
    const matchesCategory = category === "all" || loc.category === category;
    return matchesSearch && matchesCategory;
  });
};
