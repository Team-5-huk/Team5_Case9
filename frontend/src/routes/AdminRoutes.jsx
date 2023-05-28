import React from 'react';
import {Routes, Route} from 'react-router-dom';
import Redirect from '../helpers/Redirect';
import Admin from '../pages/Admin/Admin';
import ListFrames from '../pages/Admin/ListFrames/ListFrames';
import DashBoard from '../pages/Admin/DashBoard/Dashboard';
import Account from '../pages/Admin/Account/Account';
import Sections from '../pages/Admin/Sections/Sections';

const AdminRoutes = () => {
  return (
    <Routes>
        <Route path="/admin" element={<Admin />} />
        <Route path="/admin/account" element={<Account />} />
        <Route path="/admin/:object" element={<ListFrames />} />
        <Route path="/admin/:object/:frame" element={<Sections />} />
        <Route path="/admin/:object/:frame/:section" element={<DashBoard />} />
        <Route path='*' element={<Redirect to='/admin' />} />
    </Routes>
  )
}

export default AdminRoutes;