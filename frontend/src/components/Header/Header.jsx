import styles from './Header.module.scss';
import PersonOutlineIcon from '@mui/icons-material/PersonOutline';
import logo from '../../assets/logo.svg';
import { useNavigate } from 'react-router-dom';
import { observer } from 'mobx-react-lite';
import store from '../../store/store';

const Header = observer(() => {
	const { authUser } = store;
	const navigate = useNavigate();

	const handleNavigate = () => {
		const pages = {
			'': '/sigin',
			'employee': '/employee',
			'admin': '/admin',
			'user': '/client',
		};

		navigate(pages[authUser]);
	};

	const accountNavigate = () => {
		const pages = {
			'': '/sigin',
			'employee': '/employee/account',
			'admin': '/admin/account',
			'user': '/client/account',
		};

		navigate(pages[authUser]);
	};

	return (
		<div className={styles.container}>
			<img className={styles.logo} src={logo} onClick={handleNavigate} />

			<div className={styles.title}>АНАЛИЗ ГОТОВНОСТИ ВЫПОЛНЕННЫХ РАБОТ</div>

			{authUser && (
				<div className={styles.account} onClick={accountNavigate}>
					<PersonOutlineIcon />
					Аккаунт
				</div>
			)}
		</div>
	);
});

export default Header;
