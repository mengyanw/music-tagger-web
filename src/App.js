import * as React from 'react';
import { Route, Routes } from 'react-router-dom';
import ResponsiveAppBar from "./NavBar";
import HomeScreen from "./HomeScreen";
import ContactScreen from "./ContactScreen";
import ReportScreen from "./ReportScreen";
import Copyright from './CopyRight';

function App() {
  return (
    <div className="App">
      <ResponsiveAppBar />
      <Routes>
        <Route path="/" element={<HomeScreen />} />
        <Route path="/report" element={<ReportScreen />} />
        <Route path="/contact" element={<ContactScreen />} />
      </Routes>
      <Copyright />
    </div>
  );
}

export default App;
