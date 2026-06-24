"""
Digital Shield Rail Defense — Railway Station Database
========================================================
Expanded database of 50 major Indian railway stations with
real geo-coordinates, platform counts, zone/division info,
daily footfall estimates, and trafficking risk tiers.
"""

from backend.services.railway_schema import StationSchema

# ============================================================================
# 50 MAJOR INDIAN RAILWAY STATIONS
# ============================================================================

STATIONS_DB = [
    # --- NORTHERN ZONE ---
    StationSchema("NDLS", "New Delhi", "New Delhi", "Delhi", "Northern", "Delhi", 28.6425, 77.2196, 16, True, True, 450000, "HIGH"),
    StationSchema("DLI", "Old Delhi Junction", "Delhi", "Delhi", "Northern", "Delhi", 28.6566, 77.2277, 16, True, False, 200000, "HIGH"),
    StationSchema("LKO", "Lucknow Charbagh", "Lucknow", "Uttar Pradesh", "Northern", "Lucknow", 26.8314, 80.9204, 9, True, False, 150000, "ELEVATED"),
    StationSchema("CDG", "Chandigarh Junction", "Chandigarh", "Chandigarh", "Northern", "Ambala", 30.6767, 76.8092, 6, True, False, 55000, "STANDARD"),
    StationSchema("ASR", "Amritsar Junction", "Amritsar", "Punjab", "Northern", "Firozpur", 31.6340, 74.8723, 6, True, False, 60000, "ELEVATED"),
    StationSchema("UMB", "Ambala Cantt", "Ambala", "Haryana", "Northern", "Ambala", 30.3752, 76.7821, 7, True, False, 70000, "STANDARD"),
    StationSchema("JRC", "Jammu Tawi", "Jammu", "J&K", "Northern", "Firozpur", 32.7266, 74.8570, 5, False, True, 35000, "STANDARD"),
    # --- WESTERN ZONE ---
    StationSchema("BCT", "Mumbai Central", "Mumbai", "Maharashtra", "Western", "Mumbai", 18.9712, 72.8193, 11, False, True, 350000, "HIGH"),
    StationSchema("ADI", "Ahmedabad Junction", "Ahmedabad", "Gujarat", "Western", "Ahmedabad", 23.0258, 72.6004, 12, True, False, 180000, "ELEVATED"),
    StationSchema("JP", "Jaipur Junction", "Jaipur", "Rajasthan", "North Western", "Jaipur", 26.9196, 75.7878, 7, True, False, 120000, "ELEVATED"),
    StationSchema("BRC", "Vadodara Junction", "Vadodara", "Gujarat", "Western", "Vadodara", 22.3100, 73.1812, 7, True, False, 100000, "STANDARD"),
    StationSchema("ST", "Surat", "Surat", "Gujarat", "Western", "Mumbai", 21.2053, 72.8413, 5, False, False, 85000, "STANDARD"),
    StationSchema("AII", "Ajmer Junction", "Ajmer", "Rajasthan", "North Western", "Ajmer", 26.4499, 74.6400, 5, True, False, 40000, "STANDARD"),
    StationSchema("JU", "Jodhpur Junction", "Jodhpur", "Rajasthan", "North Western", "Jodhpur", 26.2889, 73.0178, 6, True, False, 45000, "STANDARD"),
    # --- CENTRAL ZONE ---
    StationSchema("CSTM", "Chhatrapati Shivaji Terminus", "Mumbai", "Maharashtra", "Central", "Mumbai", 18.9398, 72.8355, 18, True, True, 500000, "HIGH"),
    StationSchema("PUNE", "Pune Junction", "Pune", "Maharashtra", "Central", "Pune", 18.5287, 73.8745, 6, True, False, 130000, "ELEVATED"),
    StationSchema("NGP", "Nagpur Junction", "Nagpur", "Maharashtra", "Central", "Nagpur", 21.1472, 79.0845, 8, True, False, 80000, "STANDARD"),
    StationSchema("BPL", "Bhopal Junction", "Bhopal", "Madhya Pradesh", "West Central", "Bhopal", 23.2688, 77.4134, 6, True, False, 90000, "STANDARD"),
    StationSchema("JBP", "Jabalpur Junction", "Jabalpur", "Madhya Pradesh", "West Central", "Jabalpur", 23.1687, 79.9450, 6, True, False, 45000, "STANDARD"),
    StationSchema("NZM", "Hazrat Nizamuddin", "New Delhi", "Delhi", "Central", "Delhi", 28.5860, 77.2509, 7, False, False, 200000, "HIGH"),
    # --- EASTERN ZONE ---
    StationSchema("HWH", "Howrah Junction", "Kolkata", "West Bengal", "Eastern", "Howrah", 22.5836, 88.3422, 23, True, True, 600000, "HIGH"),
    StationSchema("SDAH", "Sealdah", "Kolkata", "West Bengal", "Eastern", "Sealdah", 22.5645, 88.3723, 20, True, True, 500000, "HIGH"),
    StationSchema("PNBE", "Patna Junction", "Patna", "Bihar", "East Central", "Danapur", 25.6079, 85.1001, 10, True, False, 180000, "ELEVATED"),
    StationSchema("DNR", "Danapur", "Patna", "Bihar", "East Central", "Danapur", 25.6200, 85.0491, 5, False, False, 50000, "STANDARD"),
    StationSchema("RNC", "Ranchi Junction", "Ranchi", "Jharkhand", "South Eastern", "Ranchi", 23.3143, 85.3214, 6, True, False, 55000, "STANDARD"),
    StationSchema("GHY", "Guwahati", "Guwahati", "Assam", "Northeast Frontier", "Lumding", 26.1831, 91.7504, 5, False, False, 60000, "ELEVATED"),
    StationSchema("NJP", "New Jalpaiguri", "Siliguri", "West Bengal", "Northeast Frontier", "Katihar", 26.7093, 88.4327, 8, True, False, 50000, "ELEVATED"),
    StationSchema("MGS", "Mughal Sarai Junction", "Chandauli", "Uttar Pradesh", "East Central", "Mughal Sarai", 25.2830, 83.1164, 9, True, False, 90000, "STANDARD"),
    # --- SOUTHERN ZONE ---
    StationSchema("MAS", "Chennai Central", "Chennai", "Tamil Nadu", "Southern", "Chennai", 13.0827, 80.2755, 17, True, True, 400000, "HIGH"),
    StationSchema("SBC", "KSR Bengaluru", "Bengaluru", "Karnataka", "South Western", "Bengaluru", 12.9779, 77.5661, 10, True, True, 250000, "ELEVATED"),
    StationSchema("SC", "Secunderabad Junction", "Hyderabad", "Telangana", "South Central", "Secunderabad", 17.4337, 78.5016, 10, True, False, 200000, "ELEVATED"),
    StationSchema("HYB", "Hyderabad Deccan", "Hyderabad", "Telangana", "South Central", "Hyderabad", 17.3593, 78.4703, 5, False, True, 80000, "STANDARD"),
    StationSchema("TVC", "Thiruvananthapuram Central", "Thiruvananthapuram", "Kerala", "Southern", "Thiruvananthapuram", 8.4894, 76.9507, 5, False, True, 60000, "STANDARD"),
    StationSchema("ERS", "Ernakulam Junction", "Kochi", "Kerala", "Southern", "Thiruvananthapuram", 9.9680, 76.2882, 6, True, False, 55000, "STANDARD"),
    StationSchema("CBE", "Coimbatore Junction", "Coimbatore", "Tamil Nadu", "Southern", "Salem", 11.0051, 76.9665, 6, True, False, 45000, "STANDARD"),
    StationSchema("MDU", "Madurai Junction", "Madurai", "Tamil Nadu", "Southern", "Madurai", 9.9191, 78.1174, 5, True, False, 40000, "STANDARD"),
    StationSchema("VSKP", "Visakhapatnam", "Visakhapatnam", "Andhra Pradesh", "East Coast", "Waltair", 17.7215, 83.2889, 8, True, False, 70000, "STANDARD"),
    StationSchema("BBS", "Bhubaneswar", "Bhubaneswar", "Odisha", "East Coast", "Khurda Road", 20.2713, 85.8388, 6, False, False, 55000, "STANDARD"),
    # --- NORTH CENTRAL & OTHERS ---
    StationSchema("CNB", "Kanpur Central", "Kanpur", "Uttar Pradesh", "North Central", "Kanpur", 26.4534, 80.3515, 10, True, False, 120000, "ELEVATED"),
    StationSchema("AGC", "Agra Cantt", "Agra", "Uttar Pradesh", "North Central", "Agra", 27.1628, 78.0108, 7, True, False, 85000, "STANDARD"),
    StationSchema("BSB", "Varanasi Junction", "Varanasi", "Uttar Pradesh", "Northern", "Varanasi", 25.3159, 83.0101, 9, True, False, 130000, "ELEVATED"),
    StationSchema("ALD", "Prayagraj Junction", "Prayagraj", "Uttar Pradesh", "North Central", "Prayagraj", 25.4299, 81.8436, 10, True, False, 90000, "ELEVATED"),
    StationSchema("GWL", "Gwalior Junction", "Gwalior", "Madhya Pradesh", "North Central", "Jhansi", 26.2183, 78.1828, 5, True, False, 40000, "STANDARD"),
    StationSchema("TPTY", "Tirupati", "Tirupati", "Andhra Pradesh", "South Central", "Guntakal", 13.6316, 79.4192, 6, True, False, 65000, "STANDARD"),
    StationSchema("RJT", "Rajkot Junction", "Rajkot", "Gujarat", "Western", "Rajkot", 22.2951, 70.8008, 5, True, False, 35000, "STANDARD"),
    StationSchema("UDZ", "Udaipur City", "Udaipur", "Rajasthan", "North Western", "Ajmer", 24.5854, 73.6952, 4, False, False, 25000, "STANDARD"),
    StationSchema("SVDK", "Shri Mata Vaishno Devi Katra", "Katra", "J&K", "Northern", "Firozpur", 32.9915, 74.9318, 4, False, True, 20000, "STANDARD"),
    StationSchema("ROU", "Rourkela Junction", "Rourkela", "Odisha", "South Eastern", "Chakradharpur", 22.2588, 84.8536, 5, True, False, 30000, "STANDARD"),
    StationSchema("DHN", "Dhanbad Junction", "Dhanbad", "Jharkhand", "East Central", "Dhanbad", 23.7957, 86.4304, 7, True, False, 65000, "STANDARD"),
]

# Quick lookup by code
STATION_LOOKUP = {s.code: s for s in STATIONS_DB}
