import axios from "axios";

const API_URL = "http://localhost:8000"; // Adjust if your backend port differs

// 1. Fetch Markers (Hotels, Parks, etc.)
export const fetchLocations = async (district: string, place: string) => {
  // If your backend only uses district, that's fine. 
  // We keep 'place' in the signature just in case you expand the API later.
  return axios.get(`${API_URL}/${place}`);
};

// 2. Fetch Existing Roads (The grey background network)
// You need this one so the map isn't empty behind your proposed roads!
export const fetchRoads = async (place: string) => {
  return axios.get(`${API_URL}/roads/${place}`);
};

// 3. Fetch Proposed Roads (The red/blue analysis results)
export const fetchProposedRoads = async (place: string, district: string) => {
  return axios.get(`${API_URL}/roads/${place}/${district}`);
};
