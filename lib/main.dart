import 'package:flutter/material.dart';

void main() {
  runApp(const MedusaApp());
}

class MedusaApp extends StatelessWidget {
  const MedusaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Medusa',
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF5F7F8),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF087F73),
        ),
      ),
      home: const MedusaHome(),
    );
  }
}

class MedusaHome extends StatefulWidget {
  const MedusaHome({super.key});

  @override
  State<MedusaHome> createState() => _MedusaHomeState();
}

class _MedusaHomeState extends State<MedusaHome> {
  int selectedIndex = 0;

  final List<Widget> pages = const [
    HomePage(),
    DetectPage(),
    MarketplacePage(),
    HealthPage(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: pages[selectedIndex],

      bottomNavigationBar: NavigationBar(
        selectedIndex: selectedIndex,
        onDestinationSelected: (index) {
          setState(() {
            selectedIndex = index;
          });
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.home_outlined),
            selectedIcon: Icon(Icons.home),
            label: 'Home',
          ),
          NavigationDestination(
            icon: Icon(Icons.biotech_outlined),
            selectedIcon: Icon(Icons.biotech),
            label: 'Detect',
          ),
          NavigationDestination(
            icon: Icon(Icons.shopping_bag_outlined),
            selectedIcon: Icon(Icons.shopping_bag),
            label: 'Market',
          ),
          NavigationDestination(
            icon: Icon(Icons.favorite_outline),
            selectedIcon: Icon(Icons.favorite),
            label: 'Health',
          ),
        ],
      ),
    );
  }
}

// ============================================================
// HOME
// ============================================================

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Welcome back',
                    style: TextStyle(
                      color: Colors.grey,
                      fontSize: 14,
                    ),
                  ),
                  SizedBox(height: 5),
                  Text(
                    'Medusa',
                    style: TextStyle(
                      fontSize: 30,
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                ],
              ),

              CircleAvatar(
                radius: 24,
                backgroundColor: const Color(0xFFE2F3F0),
                child: Icon(
                  Icons.person,
                  color: const Color(0xFF087F73),
                ),
              ),
            ],
          ),

          const SizedBox(height: 25),

          // HEALTH INDEX
          Container(
            padding: const EdgeInsets.all(22),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(25),
              gradient: const LinearGradient(
                colors: [
                  Color(0xFF087F73),
                  Color(0xFF16A394),
                ],
              ),
            ),
            child: Row(
              children: [
                const Expanded(
                  child: Column(
                    crossAxisAlignment:
                        CrossAxisAlignment.start,
                    children: [
                      Text(
                        'MEDUSA HEALTH INDEX',
                        style: TextStyle(
                          color: Colors.white70,
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      SizedBox(height: 8),
                      Text(
                        '78',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 52,
                          fontWeight: FontWeight.w900,
                        ),
                      ),
                      Text(
                        'Your health profile',
                        style: TextStyle(
                          color: Colors.white,
                        ),
                      ),
                    ],
                  ),
                ),

                Container(
                  width: 85,
                  height: 85,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(
                      color: Colors.white,
                      width: 6,
                    ),
                  ),
                  child: const Center(
                    child: Text(
                      '78%',
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 18,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 22),

          // AI DETECTION
          GestureDetector(
            onTap: () {
              setState(() {});
            },
            child: Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(22),
              ),
              child: const Row(
                children: [
                  CircleAvatar(
                    radius: 29,
                    backgroundColor: Color(0xFFE2F3F0),
                    child: Icon(
                      Icons.biotech,
                      color: Color(0xFF087F73),
                      size: 29,
                    ),
                  ),

                  SizedBox(width: 15),

                  Expanded(
                    child: Column(
                      crossAxisAlignment:
                          CrossAxisAlignment.start,
                      children: [
                        Text(
                          'AI Health Detection',
                          style: TextStyle(
                            fontSize: 17,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        SizedBox(height: 5),
                        Text(
                          'Analyse a medical image with Medusa AI',
                          style: TextStyle(
                            color: Colors.grey,
                            fontSize: 13,
                          ),
                        ),
                      ],
                    ),
                  ),

                  Icon(Icons.arrow_forward_ios, size: 16),
                ],
              ),
            ),
          ),

          const SizedBox(height: 28),

          const Text(
            'Quick actions',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),

          const SizedBox(height: 15),

          Row(
            children: [
              Expanded(
                child: QuickAction(
                  icon: Icons.local_hospital,
                  title: 'Find Care',
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: QuickAction(
                  icon: Icons.science,
                  title: 'Laboratory',
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: QuickAction(
                  icon: Icons.calendar_month,
                  title: 'Appointments',
                ),
              ),
            ],
          ),

          const SizedBox(height: 28),

          const Text(
            'Recent activity',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),

          const SizedBox(height: 15),

          Container(
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(18),
            ),
            child: const Row(
              children: [
                Icon(
                  Icons.history,
                  color: Color(0xFF087F73),
                ),
                SizedBox(width: 14),
                Expanded(
                  child: Text(
                    'No recent AI analysis',
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                Icon(
                  Icons.chevron_right,
                  color: Colors.grey,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================
// QUICK ACTION
// ============================================================

class QuickAction extends StatelessWidget {
  final IconData icon;
  final String title;

  const QuickAction({
    super.key,
    required this.icon,
    required this.title,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(
        vertical: 18,
        horizontal: 5,
      ),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        children: [
          Icon(
            icon,
            color: const Color(0xFF087F73),
            size: 27,
          ),
          const SizedBox(height: 8),
          Text(
            title,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================
// AI DETECTION
// ============================================================

class DetectPage extends StatelessWidget {
  const DetectPage({super.key});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          const Text(
            'AI Detection',
            style: TextStyle(
              fontSize: 29,
              fontWeight: FontWeight.w900,
            ),
          ),

          const SizedBox(height: 8),

          const Text(
            'Choose what you want Medusa to analyse.',
            style: TextStyle(
              color: Colors.grey,
              fontSize: 15,
            ),
          ),

          const SizedBox(height: 25),

          DetectionCard(
            icon: Icons.medical_information,
            title: 'Medical Imaging',
            subtitle:
                'Ultrasound, MRI, X-ray and CT',
          ),

          DetectionCard(
            icon: Icons.face,
            title: 'Skin Analysis',
            subtitle:
                'AI-assisted skin image screening',
          ),

          DetectionCard(
            icon: Icons.visibility,
            title: 'Eye Analysis',
            subtitle:
                'AI-assisted eye screening',
          ),

          DetectionCard(
            icon: Icons.monitor_heart,
            title: 'Cardiac Analysis',
            subtitle:
                'ECG and cardiovascular screening',
          ),

          const SizedBox(height: 20),

          Container(
            padding: const EdgeInsets.all(17),
            decoration: BoxDecoration(
              color: const Color(0xFFFFF5DF),
              borderRadius: BorderRadius.circular(18),
            ),
            child: const Row(
              crossAxisAlignment:
                  CrossAxisAlignment.start,
              children: [
                Icon(
                  Icons.info_outline,
                  color: Colors.orange,
                ),
                SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'Medusa provides AI-assisted screening and does not replace professional medical diagnosis.',
                    style: TextStyle(fontSize: 13),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================
// DETECTION CARD
// ============================================================

class DetectionCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;

  const DetectionCard({
    super.key,
    required this.icon,
    required this.title,
    required this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        children: [
          Container(
            width: 55,
            height: 55,
            decoration: BoxDecoration(
              color: const Color(0xFFE2F3F0),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Icon(
              icon,
              color: const Color(0xFF087F73),
            ),
          ),

          const SizedBox(width: 15),

          Expanded(
            child: Column(
              crossAxisAlignment:
                  CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
                const SizedBox(height: 5),
                Text(
                  subtitle,
                  style: const TextStyle(
                    color: Colors.grey,
                    fontSize: 13,
                  ),
                ),
              ],
            ),
          ),

          const Icon(Icons.chevron_right),
        ],
      ),
    );
  }
}

// ============================================================
// MARKETPLACE
// ============================================================

class MarketplacePage extends StatelessWidget {
  const MarketplacePage({super.key});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          const Text(
            'Marketplace',
            style: TextStyle(
              fontSize: 29,
              fontWeight: FontWeight.w900,
            ),
          ),

          const SizedBox(height: 18),

          Container(
            padding:
                const EdgeInsets.symmetric(
              horizontal: 15,
            ),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
            ),
            child: const TextField(
              decoration: InputDecoration(
                icon: Icon(Icons.search),
                hintText: 'Search healthcare...',
                border: InputBorder.none,
              ),
            ),
          ),

          const SizedBox(height: 25),

          const Text(
            'Categories',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),

          const SizedBox(height: 15),

          Wrap(
            spacing: 9,
            runSpacing: 9,
            children: const [
              MarketChip(
                icon: Icons.person,
                title: 'Doctors',
              ),
              MarketChip(
                icon: Icons.local_hospital,
                title: 'Hospitals',
              ),
              MarketChip(
                icon: Icons.science,
                title: 'Labs',
              ),
              MarketChip(
                icon: Icons.medication,
                title: 'Pharmacy',
              ),
              MarketChip(
                icon: Icons.shopping_bag,
                title: 'Products',
              ),
              MarketChip(
                icon: Icons.medical_services,
                title: 'Services',
              ),
            ],
          ),

          const SizedBox(height: 30),

          const Text(
            'Featured',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),

          const SizedBox(height: 15),

          MarketplaceItem(
            icon: Icons.science,
            title: 'Full Blood Panel',
            subtitle: 'Laboratory service',
            price: '₦25,000',
          ),

          MarketplaceItem(
            icon: Icons.person,
            title: 'Specialist Consultation',
            subtitle: 'Healthcare consultation',
            price: '₦15,000',
          ),

          MarketplaceItem(
            icon: Icons.medical_information,
            title: 'Ultrasound Scan',
            subtitle: 'Diagnostic imaging',
            price: '₦20,000',
          ),
        ],
      ),
    );
  }
}

// ============================================================
// MARKET CHIP
// ============================================================

class MarketChip extends StatelessWidget {
  final IconData icon;
  final String title;

  const MarketChip({
    super.key,
    required this.icon,
    required this.title,
  });

  @override
  Widget build(BuildContext context) {
    return Chip(
      avatar: Icon(
        icon,
        size: 18,
        color: const Color(0xFF087F73),
      ),
      label: Text(title),
      backgroundColor: Colors.white,
      side: BorderSide.none,
    );
  }
}

// ============================================================
// MARKETPLACE ITEM
// ============================================================

class MarketplaceItem extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final String price;

  const MarketplaceItem({
    super.key,
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.price,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(17),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        children: [
          Container(
            width: 58,
            height: 58,
            decoration: BoxDecoration(
              color: const Color(0xFFE2F3F0),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Icon(
              icon,
              color: const Color(0xFF087F73),
            ),
          ),

          const SizedBox(width: 14),

          Expanded(
            child: Column(
              crossAxisAlignment:
                  CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 5),
                Text(
                  subtitle,
                  style: const TextStyle(
                    color: Colors.grey,
                    fontSize: 12,
                  ),
                ),
                const SizedBox(height: 7),
                Text(
                  price,
                  style: const TextStyle(
                    color: Color(0xFF087F73),
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),

          const Icon(Icons.chevron_right),
        ],
      ),
    );
  }
}

// ============================================================
// HEALTH
// ============================================================

class HealthPage extends StatelessWidget {
  const HealthPage({super.key});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          const Text(
            'My Health',
            style: TextStyle(
              fontSize: 29,
              fontWeight: FontWeight.w900,
            ),
          ),

          const SizedBox(height: 20),

          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(22),
            ),
            child: const Column(
              crossAxisAlignment:
                  CrossAxisAlignment.start,
              children: [
                Text(
                  'Health Profile',
                  style: TextStyle(
                    fontSize: 21,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                SizedBox(height: 8),
                Text(
                  'Your health information will appear here.',
                  style: TextStyle(
                    color: Colors.grey,
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 20),

          HealthItem(
            icon: Icons.assessment,
            title: 'AI Reports',
          ),

          HealthItem(
            icon: Icons.science,
            title: 'Laboratory Results',
          ),

          HealthItem(
            icon: Icons.folder_shared,
            title: 'Medical Records',
          ),

          HealthItem(
            icon: Icons.calendar_month,
            title: 'Appointments',
          ),

          HealthItem(
            icon: Icons.medication,
            title: 'Medications',
          ),

          HealthItem(
            icon: Icons.watch,
            title: 'Health Measurements',
          ),
        ],
      ),
    );
  }
}

// ============================================================
// HEALTH ITEM
// ============================================================

class HealthItem extends StatelessWidget {
  final IconData icon;
  final String title;

  const HealthItem({
    super.key,
    required this.icon,
    required this.title,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
      ),
      child: Row(
        children: [
          Icon(
            icon,
            color: const Color(0xFF087F73),
          ),
          const SizedBox(width: 15),
          Expanded(
            child: Text(
              title,
              style: const TextStyle(
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          const Icon(
            Icons.chevron_right,
            color: Colors.grey,
          ),
        ],
      ),
    );
  }
}
