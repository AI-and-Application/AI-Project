import 'dart:async';

import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart' as gmaps;
import 'package:geolocator/geolocator.dart';
import 'package:permission_handler/permission_handler.dart';
//import 'package:record/record.dart' as rec;
import 'package:flutter_polyline_points/flutter_polyline_points.dart';
import 'package:google_maps_webservice/directions.dart' as gmd;
import 'package:google_maps_webservice/places.dart' as gmp;

void main() {
  runApp(const GoogleMapsNavigationApp());
}

class GoogleMapsNavigationApp extends StatelessWidget {
  const GoogleMapsNavigationApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'Google Maps Navigation',
      debugShowCheckedModeBanner: false,
      home: NavigationScreen(),
    );
  }
}

class NavigationScreen extends StatefulWidget {
  const NavigationScreen({super.key});

  @override
  State<NavigationScreen> createState() => _NavigationScreenState();
}

class _NavigationScreenState extends State<NavigationScreen> {
  static const String apiKey = "AIzaSyBevGewSdAnJtkl-Zyv2nD1AY2n10sFT6Q";
  gmaps.GoogleMapController? _mapController;
  gmaps.LatLng _initialPosition = const gmaps.LatLng(37.7749, -122.4194);
  gmaps.LatLng? _currentPosition;

  final gmd.GoogleMapsDirections _directions =
      gmd.GoogleMapsDirections(apiKey: apiKey);
  final gmp.GoogleMapsPlaces _places = gmp.GoogleMapsPlaces(apiKey: apiKey);

  bool _panelVisible = false;

  final TextEditingController _startController = TextEditingController();
  final TextEditingController _endController = TextEditingController();
  final TextEditingController _destController = TextEditingController();
  final TextEditingController _keywordController = TextEditingController();
  final TextEditingController _destSimController = TextEditingController();

  final Set<gmaps.Marker> _markers = {};
  final Set<gmaps.Polyline> _polylines = {};

  Timer? _instructionTimer;
  int _currentStepIndex = 0;
  List<gmd.Route> _currentRoutes = [];
  OverlayEntry? _instructionOverlay;

  bool _isRecording = false;
  //final rec.Record _recorder = rec.Record(); // Use the alias here

  StreamSubscription<Position>? _positionStream;

  @override
  void initState() {
    super.initState();
    _requestPermissions();
    _initLocation();
  }

  Future<void> _requestPermissions() async {
    await Permission.location.request();
    await Permission.microphone.request();
  }

  Future<void> _initLocation() async {
    Position position = await Geolocator.getCurrentPosition();
    setState(() {
      _currentPosition = gmaps.LatLng(position.latitude, position.longitude);
      _initialPosition = _currentPosition!;
    });
  }

  @override
  void dispose() {
    _instructionTimer?.cancel();
    _instructionOverlay?.remove();
    _positionStream?.cancel();
    super.dispose();
  }

  void _togglePanel() {
    setState(() {
      _panelVisible = !_panelVisible;
    });
  }

  Future<void> _navigateWithStartAndEnd() async {
    final start = _startController.text.trim();
    final end = _endController.text.trim();

    if (start.isEmpty || end.isEmpty) {
      _showSnackBar("출발지와 도착지를 입력하세요.");
      return;
    }

    final startCoords = await _geocodeAddress(start);
    final endCoords = await _geocodeAddress(end);

    if (startCoords == null || endCoords == null) {
      _showSnackBar("경로를 찾을 수 없습니다.");
      return;
    }

    await _calculateAndDisplayRoute(startCoords, endCoords);
  }

  Future<void> _calculateAndDisplayRoute(
      gmaps.LatLng origin, gmaps.LatLng destination) async {
    final response = await _directions.directionsWithLocation(
      gmd.Location(lat: origin.latitude, lng: origin.longitude),
      gmd.Location(lat: destination.latitude, lng: destination.longitude),
      travelMode: gmd.TravelMode.driving,
    );

    if (response.isOkay) {
      setState(() {
        _currentRoutes = response.routes;
      });
      if (_currentRoutes.isNotEmpty) {
        final route = _currentRoutes.first;
        _showRouteOnMap(route);
        _displayRouteMessages(route);
      }
    } else {
      _showSnackBar("경로를 찾을 수 없습니다.");
    }
  }

  void _showRouteOnMap(gmd.Route route) {
    _markers.clear();
    _polylines.clear();

    final leg = route.legs.first;
    // Add markers for start and end
    _markers.add(gmaps.Marker(
        markerId: const gmaps.MarkerId("start"),
        position: gmaps.LatLng(leg.startLocation.lat, leg.startLocation.lng),
        infoWindow: const gmaps.InfoWindow(title: "출발지")));
    _markers.add(gmaps.Marker(
        markerId: const gmaps.MarkerId("end"),
        position: gmaps.LatLng(leg.endLocation.lat, leg.endLocation.lng),
        infoWindow: const gmaps.InfoWindow(title: "도착지")));

    // Decode polyline
    final polyPoints = PolylinePoints();
    final decoded = polyPoints.decodePolyline(route.overviewPolyline.points);

    final gmaps.Polyline gmapsPolyline = gmaps.Polyline(
      polylineId: const gmaps.PolylineId("route"),
      points:
          decoded.map((e) => gmaps.LatLng(e.latitude, e.longitude)).toList(),
      width: 5,
    );
    _polylines.add(gmapsPolyline);

    setState(() {});
    _mapController?.animateCamera(
      gmaps.CameraUpdate.newLatLngZoom(
          gmaps.LatLng(leg.startLocation.lat, leg.startLocation.lng), 14),
    );
  }

  void _displayRouteMessages(gmd.Route route) {
    final steps = route.legs.first.steps;
    if (steps.isEmpty) return;

    _instructionTimer?.cancel();
    _instructionOverlay?.remove();
    _currentStepIndex = 0;
    _showInstructionOverlay(
        "${_sanitizeHtml(steps[_currentStepIndex].htmlInstructions)} (${steps[_currentStepIndex].distance.text})");

    _instructionTimer = Timer.periodic(const Duration(seconds: 5), (timer) {
      _currentStepIndex++;
      if (_currentStepIndex < steps.length) {
        _showInstructionOverlay(
            "${_sanitizeHtml(steps[_currentStepIndex].htmlInstructions)} (${steps[_currentStepIndex].distance.text})");
      } else {
        timer.cancel();
        _showInstructionOverlay("도착지에 도달했습니다!");
      }
    });
  }

  String _sanitizeHtml(String html) {
    final exp = RegExp(r"<[^>]*>", multiLine: true, caseSensitive: false);
    return html.replaceAll(exp, '');
  }

  void _showInstructionOverlay(String instruction) {
    _instructionOverlay?.remove();
    _instructionOverlay = _createOverlay(instruction);
    Overlay.of(context).insert(_instructionOverlay!);
  }

  OverlayEntry _createOverlay(String text) {
    return OverlayEntry(builder: (context) {
      return Positioned(
        bottom: 20,
        left: MediaQuery.of(context).size.width * 0.5 - 150,
        child: Material(
          color: Colors.transparent,
          child: Container(
            width: 300,
            padding: const EdgeInsets.all(15),
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.7),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              text,
              style: const TextStyle(color: Colors.white, fontSize: 16),
              textAlign: TextAlign.center,
            ),
          ),
        ),
      );
    });
  }

  Future<gmaps.LatLng?> _geocodeAddress(String address) async {
    final geocodeResponse = await _places.searchByText(address);
    if (geocodeResponse.isOkay && geocodeResponse.results.isNotEmpty) {
      final loc = geocodeResponse.results.first.geometry!.location;
      return gmaps.LatLng(loc.lat, loc.lng);
    }
    return null;
  }

  Future<void> _startRealTimeNavigation() async {
    final destination = _destController.text.trim();
    if (destination.isEmpty) {
      _showSnackBar("도착지를 입력하세요.");
      return;
    }
    final destCoords = await _geocodeAddress(destination);
    if (destCoords == null) {
      _showSnackBar("주소를 찾을 수 없습니다.");
      return;
    }

    _positionStream?.cancel();
    _positionStream = Geolocator.getPositionStream(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.high,
        distanceFilter: 0,
      ),
    ).listen((position) {
      final userLoc = gmaps.LatLng(position.latitude, position.longitude);
      _updateUserMarker(userLoc);
      _mapController
          ?.animateCamera(gmaps.CameraUpdate.newLatLngZoom(userLoc, 16));
      _calculateAndDisplayRoute(userLoc, destCoords);
    });
  }

  void _stopRealTimeNavigation() {
    if (_positionStream != null) {
      _positionStream!.cancel();
      _positionStream = null;
    }
    _showSnackBar("실시간 내비게이션이 중지되었습니다.");
  }

  Future<void> _searchPOI() async {
    final keyword = _keywordController.text.trim();
    if (keyword.isEmpty) {
      _showSnackBar("키워드를 입력하세요.");
      return;
    }

    final placesResponse = await _places.searchByText(keyword);
    if (placesResponse.isOkay && placesResponse.results.isNotEmpty) {
      final place = placesResponse.results.first;
      final loc = place.geometry!.location;
      final poiLoc = gmaps.LatLng(loc.lat, loc.lng);
      _mapController
          ?.animateCamera(gmaps.CameraUpdate.newLatLngZoom(poiLoc, 16));

      setState(() {
        _markers.add(gmaps.Marker(
            markerId: const gmaps.MarkerId("poi"),
            position: poiLoc,
            infoWindow: gmaps.InfoWindow(title: place.name)));
      });
    } else {
      _showSnackBar("POI를 찾을 수 없습니다.");
    }
  }

  Future<void> _startSimulation() async {
    final destination = _destSimController.text.trim();
    if (destination.isEmpty) {
      _showSnackBar("도착지를 입력하세요.");
      return;
    }

    final destCoords = await _geocodeAddress(destination);
    if (destCoords == null) {
      _showSnackBar("주소를 찾을 수 없습니다.");
      return;
    }

    gmaps.LatLng simulatedLocation = _initialPosition;
    Timer.periodic(const Duration(seconds: 2), (timer) {
      simulatedLocation = gmaps.LatLng(simulatedLocation.latitude + 0.0001,
          simulatedLocation.longitude + 0.0001);

      _updateUserMarker(simulatedLocation);
      _calculateAndDisplayRoute(simulatedLocation, destCoords);

      if ((simulatedLocation.latitude - destCoords.latitude).abs() < 0.0001 &&
          (simulatedLocation.longitude - destCoords.longitude).abs() < 0.0001) {
        timer.cancel();
        _showSnackBar("시뮬레이션 완료");
      }
    });
  }

  void _updateUserMarker(gmaps.LatLng position) {
    setState(() {
      _markers.removeWhere((m) => m.markerId == const gmaps.MarkerId("user"));
      _markers.add(gmaps.Marker(
          markerId: const gmaps.MarkerId("user"),
          position: position,
          icon: gmaps.BitmapDescriptor.defaultMarkerWithHue(
              gmaps.BitmapDescriptor.hueBlue),
          infoWindow: const gmaps.InfoWindow(title: "현재 위치")));
    });
  }

  void _showSnackBar(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

/*
  Future<void> _toggleRecording() async {
    if (_isRecording) {
      final path = await _recorder.stop();
      setState(() {
        _isRecording = false;
      });
      if (path != null) {
        _showSnackBar("녹음 완료. 파일: $path");
      }
    } else {
      if (await _recorder.hasPermission()) {
        Directory appDir = await getApplicationDocumentsDirectory();
        String filePath =
            '${appDir.path}/recording_${DateTime.now().millisecondsSinceEpoch}.wav';
        await _recorder.start(path: filePath);
        setState(() {
          _isRecording = true;
        });
      } else {
        _showSnackBar("오디오 권한이 필요합니다.");
      }
    }
  }
*/
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          gmaps.GoogleMap(
            initialCameraPosition: const gmaps.CameraPosition(
                target: gmaps.LatLng(37.7749, -122.4194), zoom: 16),
            onMapCreated: (c) => _mapController = c,
            markers: _markers,
            polylines: _polylines,
            myLocationButtonEnabled: false,
            myLocationEnabled: false,
          ),
          Positioned(
            top: 10,
            left: 10,
            child: InkWell(
              onTap: _togglePanel,
              child: Container(
                width: 50,
                height: 50,
                decoration: BoxDecoration(
                  color: const Color(0xFF1e40af),
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                        color: Colors.black.withOpacity(0.1), blurRadius: 4)
                  ],
                ),
                child: const Center(
                  child: Text(
                    "☰",
                    style: TextStyle(color: Colors.white, fontSize: 24),
                  ),
                ),
              ),
            ),
          ),
          if (_panelVisible) _buildControlPanel(),
        ],
      ),
    );
  }

  Widget _buildControlPanel() {
    return Positioned(
      top: 70,
      left: 10,
      child: Container(
        width: 280,
        padding: const EdgeInsets.all(15),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.95),
          borderRadius: BorderRadius.circular(8),
          boxShadow: [
            BoxShadow(
                color: Colors.black.withOpacity(0.1),
                blurRadius: 6,
                offset: const Offset(0, 4))
          ],
        ),
        child: SingleChildScrollView(
          child: Column(
            children: [
              const Text("Google Maps Navigation",
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              const SizedBox(height: 10),
              TextField(
                controller: _startController,
                decoration: InputDecoration(
                  hintText: "출발지",
                  border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(5)),
                ),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: _endController,
                decoration: InputDecoration(
                  hintText: "도착지",
                  border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(5)),
                ),
              ),
              const SizedBox(height: 10),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF1e40af),
                ),
                onPressed: () {
                  setState(() {
                    _panelVisible = false;
                  });
                  _navigateWithStartAndEnd();
                },
                child: const Text("출발지 및 도착지 경로"),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: _destController,
                decoration: InputDecoration(
                  hintText: "도착지 (현재 위치 출발)",
                  border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(5)),
                ),
              ),
              const SizedBox(height: 10),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF1e40af),
                ),
                onPressed: () {
                  setState(() {
                    _panelVisible = false;
                  });
                  _startRealTimeNavigation();
                },
                child: const Text("실시간 내비게이션"),
              ),
              const SizedBox(height: 10),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFFF59E0B),
                ),
                onPressed: () {
                  setState(() {
                    _panelVisible = false;
                  });
                  _stopRealTimeNavigation();
                },
                child: const Text("내비게이션 중지"),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: _keywordController,
                decoration: InputDecoration(
                  hintText: "POI 검색 (예: 레스토랑)",
                  border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(5)),
                ),
              ),
              const SizedBox(height: 10),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFFF59E0B),
                ),
                onPressed: () {
                  setState(() {
                    _panelVisible = false;
                  });
                  _searchPOI();
                },
                child: const Text("POI 검색"),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: _destSimController,
                decoration: InputDecoration(
                  hintText: "도착지 (시뮬레이션)",
                  border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(5)),
                ),
              ),
              const SizedBox(height: 10),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF1e40af),
                ),
                onPressed: () {
                  setState(() {
                    _panelVisible = false;
                  });
                  _startSimulation();
                },
                child: const Text("시뮬레이션 시작"),
              ),
              const SizedBox(height: 10),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFFF59E0B),
                ),
                onPressed: () {
                  //_toggleRecording();
                },
                child: Text(_isRecording ? "녹음 중지" : "녹음 시작"),
              ),
              const SizedBox(height: 10),
              if (!_isRecording)
                const Text(
                  "녹음 파일은 앱 문서 디렉토리에 저장됩니다.",
                  textAlign: TextAlign.center,
                ),
            ],
          ),
        ),
      ),
    );
  }
}
