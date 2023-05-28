import React from 'react'
import {Routes, Route} from 'react-router-dom';
import Redirect from '../helpers/Redirect';
import Client from '../pages/Client/Client';
import Account from '../pages/Client/Account/Account';
import InfoJC from '../pages/Client/InfoJC/InfoJC';

const ClientRoutes = () => {
  return (
    <Routes>
        <Route path="/client" element={<Client />} />
        <Route path="/client/account" element={<Account />} />
        <Route path="/client/:id" element={<InfoJC />} />
        <Route path='*' element={<Redirect to='/client' />} />
    </Routes>
  )
}

export default ClientRoutes;