import React, { useEffect, useState } from 'react';
import { apiGetUser } from '../../../api/api';
import linksStore from '../../../store/linksStore';
import CustomButton from '../../../components/CustomButton/CustomButton';
import store from '../../../store/store';

const Account = () => {
	const { setAuthUser } = store;
	const { linkGetUser } = linksStore;

	const [user, setUser] = useState({});

	useEffect(() => {
		getInfoUser();
	}, []);

	const getInfoUser = () => {
		apiGetUser(linkGetUser).then(({ data, error }) => {
			setUser(data);
			console.log(error);
		});
	};

	const signOut = () => {
		localStorage.removeItem('token');
		localStorage.removeItem('role');
		setAuthUser('logout');
	};

	return (
		<div>
			 <div>
				{user.first_name} {user.last_name}
				{user.email}
				<CustomButton name={'Выйти'} width={'80px'} handleClick={signOut} />
			</div>
			
		</div>
	);
};

export default Account;
